import os 
import numpy as np 
import torch 
import math 
from torch.optim import SGD
from tqdm.auto import tqdm
from contextlib import nullcontext

import sublora.nn.projectors as projectors 
import sublora.bounds.quantize_fns as quantize
from sublora.utils import get_batch

def sum_k_elements(row, k_th):
    return torch.sum(row[:k_th+1])


def bound_metrics_from_logits(logits, Y, device):
    # === for each token, log top-k indice, percentile, and log probability === #
    softmax_matrix = torch.nn.functional.softmax(logits,dim=-1)

    logits = None # save memory

    sorted_softmax_matrix, indices_softmax_matrix = torch.sort(softmax_matrix, dim=-1, descending=True)
        
    # 1. top-k indices
    top_k_indices = torch.argmax((indices_softmax_matrix == Y.to(device).unsqueeze(1)).long(),dim=-1)
    
    # 2. probabilities 
    selected_prob_scores = softmax_matrix[torch.arange(softmax_matrix.shape[0]), Y.to(device).view(-1)]
    
    # 3. percentile
    percentile_vec = torch.zeros((sorted_softmax_matrix.shape[0],), device=device)
    for i in range(sorted_softmax_matrix.shape[0]):
        percentile_vec[i] = sum_k_elements(sorted_softmax_matrix[i], top_k_indices[i])
    
    return top_k_indices, percentile_vec, selected_prob_scores


@torch.no_grad()
def compute_bound_scores(model, X, Y, device, intrinsic_dim, block_size, sliding_window_size, ctx):
    '''
    compute the bound metrics for a single document (i.e. assuming we don't need attention mask)
    '''

    # override ctx to be nullcontext() if we are doing subspace projection
    if intrinsic_dim != 0:
        ctx = nullcontext()

    # compute bounds metrics
    if X.shape[1] > block_size:
        with ctx:
            logits, loss = model(X[:,:block_size], Y[:,:block_size])
        logits = logits.view(-1, logits.size(-1))
        curr_Y = Y[:,:block_size].view(-1)

        # compute bounds metrics for the first 1024 tokens
        top_k_indices, percentile_vec, selected_prob_scores = bound_metrics_from_logits(logits,curr_Y,device)

        total_forward_iters = math.ceil((X.shape[1]-block_size) / sliding_window_size)
        print(f"total i's = {total_forward_iters}")
        for i in range(total_forward_iters):
            print(f"[i/total i's]=[{i}/{total_forward_iters}]")
            with ctx:
                shift_logits, _ = model(X[:,(i+1)*sliding_window_size:block_size+(i+1)*sliding_window_size], Y[:,(i+1)*sliding_window_size:block_size+(i+1)*sliding_window_size])
            shift_logits = shift_logits[:,block_size-sliding_window_size:,:]
            shift_logits = shift_logits.view(-1, logits.size(-1))
            curr_Y = Y[:,(i+1)*sliding_window_size:block_size+(i+1)*sliding_window_size].view(-1)
            
            if i == (total_forward_iters - 1):
                curr_Y = curr_Y[-shift_logits.shape[0]:]
            else:
                curr_Y = curr_Y[-sliding_window_size:]
            curr_top_k_indices, curr_percentile_vec, curr_selected_prob_scores = bound_metrics_from_logits(shift_logits, curr_Y, device)

            # update bound metrics with values from current iterations
            top_k_indices = torch.concat((top_k_indices, curr_top_k_indices))
            percentile_vec = torch.concat((percentile_vec, curr_percentile_vec))
            selected_prob_scores = torch.concat((selected_prob_scores, curr_selected_prob_scores))            
    else:
        with ctx:
            logits, loss = model(X, Y)

        logits = logits.view(-1, logits.size(-1))
        Y = Y.view(-1)
        top_k_indices, percentile_vec, selected_prob_scores = bound_metrics_from_logits(logits, Y, device)
        
    return top_k_indices, percentile_vec, selected_prob_scores
    
def compute_bound_metrics(metrics_dict, top_k_indices, selected_prob_scores, alpha_array, bound_type, eval_batch_size,
                          vocab_size, len_x):
    
    # compute local batch accuracy
    unique_indices, indices_counts = torch.unique(torch.tensor(top_k_indices),return_counts=True)
    
    if bound_type == "document_level":
        local_batch_size = 1
        
    elif bound_type == "sequence_level":
        local_batch_size = eval_batch_size 
    
    else:
        raise NotImplemented
    
    def return_avg_acc(top_acc, total_length):
        if isinstance(top_acc, int):
            avg = top_acc / total_length
        elif torch.is_tensor(top_acc) and top_acc.dim() == 0:  # Check if it's a 0-dimensional tensor (scalar)
            avg = top_acc.item() / total_length
        else:
            # Handle other cases or raise an exception if needed
            raise ValueError("Unsupported type for this top k accuracy")
        return avg 
        
    ### getting the metrics
    for k in range(1,10+1): 
        top_k_acc = 0
        i = 0 
        while unique_indices[i] < k:
            top_k_acc += indices_counts[i]
            i += 1
        # sum of accuracy over batch size
        top_k_acc = return_avg_acc(top_k_acc, len_x)
        metrics_dict[f'top_{k}_acc'] = (metrics_dict[f'top_{k}_acc'] * metrics_dict["n_train"] + top_k_acc) / (metrics_dict["n_train"] + local_batch_size)
     
    top_50_acc = 0 
    i = 0
    while unique_indices[i] < 50:
        top_50_acc += indices_counts[i] 
        i += 1 
    top_50_acc = return_avg_acc(top_50_acc, len_x)
    metrics_dict[f'top_50_acc'] = (metrics_dict[f'top_50_acc'] * metrics_dict["n_train"] + top_50_acc) / (metrics_dict["n_train"] + local_batch_size)
    
    top_100_acc = 0
    i = 0
    while unique_indices[i] < 100:
        top_100_acc += indices_counts[i] 
        i += 1 
    top_100_acc = return_avg_acc(top_100_acc, len_x)
    metrics_dict[f'top_100_acc'] = (metrics_dict[f'top_100_acc'] * metrics_dict["n_train"] + top_100_acc) / (metrics_dict["n_train"] + local_batch_size)
            
    for alpha in alpha_array:          
        log_probs = [np.log2((1-alpha)*x.item() + alpha/vocab_size) for x in selected_prob_scores]
        bdp_alpha = - sum(log_probs)/len_x

        metrics_dict[f'bpd_alpha_{alpha}'] = float((metrics_dict[f'bpd_alpha_{alpha}'] * metrics_dict["n_train"] + bdp_alpha) / (metrics_dict["n_train"] + local_batch_size))

    # update batch size estimation
    metrics_dict["n_train"] += local_batch_size
    metrics_dict["curr_iter_i"] += 1 
    
    return metrics_dict 
    

def quantize_model(model, train_data, block_size, intrinsic_dim, device_type, device, ddp, perturb_word_order_window_size,
                   quant_batch_size, max_quant_iters, use_kmeans, levels, quant_lr):
    
    if max_quant_iters > 0 and intrinsic_dim > 0:
        vector = model.subspace_params.cpu().data.numpy()
        cluster_fn = quantize.get_random_symbols_and_codebook
        if use_kmeans:
            cluster_fn = quantize.get_kmeans_symbols_and_codebook
        _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
        centroids = torch.tensor(centroids, dtype=torch.float32)
        centroids = centroids.to(device)
        quantizer_fn = quantize.Quantize().apply
        qw = quantize.QuantizingWrapper(model, quantizer=quantizer_fn, centroids=centroids)
        optim = SGD(
            [qw.subspace_params, qw.centroids],
            lr = quant_lr, momentum=0.9)

        for e in tqdm(range(max_quant_iters)):
            qw.train()
            optim.zero_grad()
            X, Y, ix = get_batch('train', train_data, None, quant_batch_size, block_size,
                                     device_type, device, perturb_word_order_window_size)
            logits, loss = qw(X, Y)
            loss.backward()
            optim.step()
            if e % 10 == 0:
                metrics = {"iter": e, "ix": ix, "mini_loss": loss.detach().item()}
                print(metrics)
        quantized_vec = qw.quantizer(qw.subspace_params, qw.centroids)
        quantized_vec = quantized_vec.cpu().detach().numpy()
        vec = (qw.centroids.unsqueeze(-2) - qw.subspace_params.unsqueeze(-1))**2.0
        symbols = torch.min(vec, -1)[-1]
        symbols = symbols.cpu().detach().numpy()
        centroids = qw.centroids.cpu().detach().numpy()
        probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
        _, coded_symbols_size = quantize.do_arithmetic_encoding(symbols, probabilities,
                                                    qw.centroids.shape[0])
        message_len = quantize.get_message_len(
            coded_symbols_size=coded_symbols_size,
            codebook=centroids,
            max_count=len(symbols),
        )
    else:
        if intrinsic_dim > 0:
            module = model.module if isinstance(model,
                                            torch.nn.parallel.DistributedDataParallel) else model
            vector = module.subspace_params.cpu().data.numpy()
            quantized_vec, message_len = quantize.quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
        else:
            aux = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            names, vector = zip(*aux)
            fvector = projectors.flatten(vector).cpu().data.numpy()
            quantized_vec, message_len = quantize.quantize_vector(fvector, levels=levels, use_kmeans=use_kmeans)
            ## free memory 
            fvector = None 

    if intrinsic_dim > 0:
        module = model.module if ddp else model
        module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)
    else:
        unfquantized_vec = projectors.unflatten_like(torch.tensor(quantized_vec), vector)
        ## free memory  
        quantized_vec, vector = None, None
        for n, p in model.named_parameters():
            for name, quantp in zip(names, unfquantized_vec):
                if n == name:
                    p.data = torch.tensor(quantp).float().to(device)
            
    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    
    return model, prefix_message_len


def get_extra_bits(intrinsic_dim, attention_linear_lora_r):
    if intrinsic_dim == 0:
        if attention_linear_lora_r == 0:
            # no_lora_no_id
            misc_extra_bits = np.ceil(np.log2(2*3))
        else:
            # lora_no_id 
            misc_extra_bits = np.ceil(np.log2(2*2*3)) 
            
    else:
        if attention_linear_lora_r == 0:
            ## no_lora_id
            misc_extra_bits = np.ceil(np.log2(2*6*3)) 
        else:
            ## lora_id
            misc_extra_bits = np.ceil(np.log2(4*6*2*3)) 
            
    return misc_extra_bits 