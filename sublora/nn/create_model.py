import os 
import pickle 
import torch 

import loralib as lora

from sublora.nn.model import GPTConfig, GPT
from sublora.nn.projectors import create_intrinsic_model

def get_model(n_layer, n_head, n_embd, bias, dropout, use_mergedlinear, apply_rope, use_mistral_sliding_window, 
              use_lora, lora_alpha, lora_dropout, attention_linear_use_lora, attention_linear_lora_r,linear_head_lora_r, 
              linear_head_enable_lora, intrinsic_dim, block_size, data_dir, out_dir, init_from, master_process, device,
              best_checkpoint_path):
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, bias=bias, vocab_size=None, dropout=dropout, 
                          use_mergedlinear=use_mergedlinear, apply_rope=apply_rope, use_lora=use_lora, lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout, attention_linear_use_lora=attention_linear_use_lora, block_size=block_size,
                          attention_linear_lora_r=attention_linear_lora_r, linear_head_lora_r=linear_head_lora_r, 
                          linear_head_enable_lora=linear_head_enable_lora, use_mistral_sliding_window=use_mistral_sliding_window)

        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            
            nparams = int(model.get_num_params())
            
            if master_process:
                torch.save(model.state_dict(), os.path.join(out_dir, 'forward_ckpt_at_random_initialization.pt'))
            
            print("INTRINSIC DIM IS: ", intrinsic_dim)
            
            if block_size < model.config.block_size:
                model._forward_net[0].crop_block_size(block_size)
                model_args['block_size'] = block_size # so that the checkpoint will have the right value
            
            if intrinsic_dim > 0:
                if use_lora:
                    lora.mark_only_lora_as_trainable(model)
                model = create_intrinsic_model(base_net=model, ckpt_path=None, intrinsic_mode="rdkronqr", intrinsic_dim=intrinsic_dim,
                                               seed=137, device=device)
                
            # crop down the model block size if desired, using model surgery
            if intrinsic_dim == 0:
                if block_size < model.config.block_size:
                    model._forward_net[0].crop_block_size(block_size)
                    model_args['block_size'] = block_size # so that the checkpoint will have the right value

            if master_process:
                torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt_at_random_initialization.pt'))
                if intrinsic_dim > 0:
                    torch.save(model.trainable_initparams, os.path.join(out_dir, 'trainable_initparams.pt'))
                    torch.save(model.names, os.path.join(out_dir, 'names.pt'))
                    
        elif init_from == 'best_ckpt':
            print(f"loading best training checkpoint from {best_checkpoint_path} for pretraining bound metrics eval") 
            ckpt_path = os.path.join(best_checkpoint_path, "best_ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_lora', 'lora_alpha', 'lora_dropout',
                      'attention_linear_use_lora', 'attention_linear_lora_r', 'linear_head_lora_r', 'linear_head_enable_lora']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            
            nparams = int(model.get_num_params())
            
            if use_lora: 
                lora.mark_only_lora_as_trainable(model)
                            
            if intrinsic_dim > 0:
                #### loading the random initialization of all the weights 
                init_ckpt_path = os.path.join(best_checkpoint_path, "forward_ckpt_at_random_initialization.pt")
                init_checkpoint = torch.load(init_ckpt_path, map_location=device)
                unwanted_prefix = '_orig_mod.'
                for k,v in list(init_checkpoint.items()):
                    if k.startswith(unwanted_prefix):
                        init_checkpoint[k[len(unwanted_prefix):]] = init_checkpoint.pop(k)

                model.load_state_dict(init_checkpoint)
                
                model = create_intrinsic_model(base_net=model,
                                            ckpt_path=None,
                                            intrinsic_mode="rdkronqr",
                                            intrinsic_dim=intrinsic_dim,
                                            seed=137,
                                            device=device,)

            state_dict = checkpoint['raw_model']
            
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    
            if intrinsic_dim > 0:
                model.load_state_dict(state_dict)
                print('subspace_params loaded!')

                model.trainable_initparams = torch.load(os.path.join(best_checkpoint_path, "trainable_initparams.pt"), map_location=device)
                model.names = torch.load(os.path.join(best_checkpoint_path, "names.pt"))
                
            else:
                model.load_state_dict(state_dict)
                
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            
            checkpoint = None # free up memory
        
        elif init_from == "openai-community/gpt2":
            model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
            iter_num, best_val_loss, model_args, nparams = None, None, None, None
        else:
            raise NotImplemented
        
        model.to(device)
        
        return model, iter_num, best_val_loss, model_args, nparams