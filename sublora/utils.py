import os
import torch 
import numpy as np 
import math 

def get_doc_indices(train_data, eot_token, openwebtext_train_eot_indices_file='/scratch/yk2516/repos/PAC_Bayes/PAC_Bayes_LoRA/nanoGPT/data/openwebtext/openwebtext_train_eot_indices_file_full.npy',
    empirical_document_length_distribution_file='/scratch/yk2516/repos/PAC_Bayes/PAC_Bayes_LoRA/nanoGPT/data/openwebtext/empirical_document_length_distribution_full.npy'):
    """ Args:
    train_data: file that includes the entire dataset. See data/prepare.py to prepare it for OpenWebText
    eot_token: the ID for the end of text token for this dataset
    openwebtext_train_eot_indices_file: path for the array that contains the IDs for EOD tokens after
        each document in the train dataset. If it's not already created, this script takes care of creating 
        it and saving it.
    empirical_document_length_distribution_file: path for the array that contains the lengths of all 
        documents in the train dataset. If it's not already created, this script takes care of creating 
        it and saving it.
    """

    if os.path.exists(openwebtext_train_eot_indices_file):
        openwebtext_train_eot_indices = np.load(openwebtext_train_eot_indices_file)
        empirical_document_length_distribution = np.load(empirical_document_length_distribution_file)
    else:
        # openwebtext_train_eot_indices
        openwebtext_train_eot_indices =  np.where(train_data==eot_token)
        openwebtext_train_eot_indices = openwebtext_train_eot_indices[0]
        openwebtext_train_eot_indices_shift_left_by_1 = np.insert(openwebtext_train_eot_indices[:-1], 0, 0)
        # empirical length distribution
        empirical_document_length_distribution = openwebtext_train_eot_indices - openwebtext_train_eot_indices_shift_left_by_1

        with open(openwebtext_train_eot_indices_file, 'wb') as f_openwebtext_train_eot_indices_file: 
            np.save(f_openwebtext_train_eot_indices_file, openwebtext_train_eot_indices)
        
        with open(empirical_document_length_distribution_file, 'wb') as f_empirical_document_length_distribution:
            np.save(f_empirical_document_length_distribution, empirical_document_length_distribution)

        openwebtext_train_eot_indices = openwebtext_train_eot_indices[:-1]
        empirical_document_length_distribution = empirical_document_length_distribution[1:] 
            
    return openwebtext_train_eot_indices, empirical_document_length_distribution

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters: 
        return learning_rate * it / warmup_iters 
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(split, train_data, val_data, batch_size, block_size, device_type, device, perturb_word_order_window_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    newx = torch.stack([torch.from_numpy((data[i:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if perturb_word_order_window_size > 1:
        for i, batch_i in enumerate(newx):
            if perturb_word_order_window_size==1024:
                # 100% shuffling
                newx[i] = newx[i][torch.randperm(len(newx[i]))]
            elif perturb_word_order_window_size < 1024:
                num_of_windows = block_size // perturb_word_order_window_size
                counter_i = 0
                while counter_i < num_of_windows:
                    sequence_segment = newx[i][counter_i*perturb_word_order_window_size:(counter_i+1)*perturb_word_order_window_size]
                    shuffled_indices = torch.randperm(perturb_word_order_window_size)
                    shuffled_sequence_segment = sequence_segment[shuffled_indices]
                    newx[i][counter_i*perturb_word_order_window_size:(counter_i+1)*perturb_word_order_window_size] = shuffled_sequence_segment
                    counter_i += 1
            else:
                raise ValueError
    x = newx[:,:-1]
    y = newx[:,1:]
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix

def sample_single_document(split, train_data, val_data, eot_token, device_type, device, 
                    openwebtext_train_eot_indices_file, empirical_document_length_distribution_file):
    '''
    This function is used for bounds evaluation where we're sampling a single document `doc_i` at a time to get log p(`doc_i`)
    '''
    
    openwebtext_train_eot_indices, empirical_document_length_distribution = get_doc_indices(train_data,
                            eot_token, 
                            openwebtext_train_eot_indices_file, 
                            empirical_document_length_distribution_file,
                            )
    
    # specify data split
    data = train_data if split == 'train' else val_data

    # sample a random document from openwebtext with replacement
    random_iter = np.random.randint(0, int((len(openwebtext_train_eot_indices))))
        
    # get document start and end index & document length
    ix = (openwebtext_train_eot_indices[random_iter]-empirical_document_length_distribution[random_iter], openwebtext_train_eot_indices[random_iter])
    length_ix = empirical_document_length_distribution[random_iter]

    # x = torch.from_numpy((data[ix[0]+1:ix[0]+length_ix]).astype(np.int64)) 
    # y = torch.from_numpy((data[ix[0]+1+1:ix[0]+length_ix+1]).astype(np.int64))

    # start from EOT token
    x = torch.from_numpy((data[ix[0]:ix[0]+length_ix-1]).astype(np.int64)) 
    y = torch.from_numpy((data[ix[0]+1:ix[0]+length_ix]).astype(np.int64))
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x.unsqueeze(0), y.unsqueeze(0), torch.tensor(ix)

def sample_nonoverlapping_sequences(split, train_data, val_data, batch_size, block_size, device_type, device, data_size):
    
    upper_bound = (data_size-1)//block_size
    lower_bound = 0
    chunk_idx = np.random.randint(lower_bound, upper_bound, size=(batch_size))
        
    data = train_data if split == 'train' else val_data
    ix = (chunk_idx[:,None]*block_size+np.arange(block_size)) # a (bs, block_size) set of ids
    x = torch.from_numpy((data[ix]).astype(np.int64)) # assuming the broadcasting is correct
    y = torch.from_numpy((data[ix+1]).astype(np.int64))
        
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix


def get_model_config(model_size):
    if model_size == "small":
        n_layer = 12
        n_head = 12
        n_embd = 768
        
    elif model_size == "small-medium":
        n_layer = 16
        n_head = 16
        n_embd = 1024
        
    elif model_size == "medium":
        n_layer = 24
        n_head = 16
        n_embd = 1024
        
    elif model_size == "medium-large":
        n_layer = 20
        n_head = 20
        n_embd = 1280
        
    elif model_size == "large":
        n_layer = 36
        n_head = 20
        n_embd = 1280
        
    elif model_size == "large-vlarge":
        n_layer = 25
        n_head = 25
        n_embd = 1600

    elif model_size == "vlarge":
        n_layer = 48
        n_head = 25
        n_embd = 1600
    else:
        pass 

    return n_layer, n_head, n_embd