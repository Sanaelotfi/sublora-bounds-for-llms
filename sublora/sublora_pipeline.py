### Code partially inspired from https://github.com/karpathy/nanoGPT 
import os
import time
import datetime
from contextlib import nullcontext
import wandb
import numpy as np
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F

from fastargs.decorators import param
from fastargs import Param, Section

import loralib as lora
from tqdm import tqdm

from sublora.nn.create_model import get_model
from sublora.bounds.bound_utils import quantize_model, compute_bound_scores, compute_bound_metrics
from sublora.bounds.compute_bounds import llm_subsampling_bound 
from sublora.utils import get_lr, get_batch, sample_single_document, sample_nonoverlapping_sequences, get_model_config
import sublora.bounds.quantize_fns as quantize

from datasets import load_dataset
from transformers import AutoTokenizer, PretrainedConfig, default_data_collator, DataCollatorWithPadding, AutoConfig, AutoModelForSequenceClassification, GPT2ForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sublora.nn.projectors import create_intrinsic_model
from torch.utils.data.dataloader import DataLoader

Section("training", "training details").params(
    gradient_accumulation_steps=Param(int, "used to simulate larger batch sizes", default=40),
    backend=Param(str, "ddp setting; 'nccl', 'gloo', etc.", default='nccl'),
    eval_interval=Param(int, "", default=500),
    log_interval=Param(int, "", default=1),
    eval_iters=Param(int, "", default=200),
    eval_only=Param(bool, "if True, script exits right after the first eval", default=False),
    always_save_checkpoint=Param(bool, "if True, always save a checkpoint after each eval", default=True),
    max_iters=Param(int, "total number of training iterations", default=600000),
)

Section("login", "login details").params(
    wandb_log=Param(bool, "disabled by default", default=False),
    wandb_project=Param(str, "name of the project", default='gpt-2'),
    wandb_run_name=Param(str, "name of the run", default='train'),
    out_dir=Param(str, "where to save results?", default='out'),
    create_new_output_dir=Param(bool, "default is True", default='True'),
)

Section("data", "data details").params(
    dataset_dir=Param(str, "name of the dataset", default="TOADD"),
    dataset=Param(str, "where to find the dataset?", default='openwebtext'),
    batch_size=Param(int, "f gradient_accumulation_steps > 1, this is the micro-batch size", default=12),
    block_size=Param(int, "size of the sequence", default=1024),
    vocab_size=Param(int, "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)", default=50257),
    data_size=Param(int, "Number of sequences in the dataset of size = block_size", default=8823811),
    num_docs=Param(int, "Number of documents in the dataset", default=8009762),
    eot_token=Param(int, "Set EOT tokens for identifying a single document within openwebtext. {'<|endoftext|>': 50256}", default=50256),
    perturb_word_order_window_size=Param(int, "perturbations window within which we apply random permutations", default=0),
    openwebtext_train_eot_indices_file=Param(str, "numpy array that contains the indices of eot tokens that separate documents", default="TOADD"),
    empirical_document_length_distribution_file=Param(str, "numpy array that contains the lengths of the different documents that constitute the train data", default="TOADD"),
)

Section("model", "model details").params(
    n_layer=Param(int, "", default=12),
    n_head=Param(int, "", default=12),
    n_embd=Param(int, "", default=768),
    dropout=Param(float, "for pretraining 0 is good, for finetuning try 0.1+", default=0.0),
    bias=Param(bool, "do we use bias inside LayerNorm and Linear layers?", default=False),
    use_mergedlinear=Param(bool, "merged linear or linear for the attention layers?", default=False),
    apply_rope=Param(bool, "apply rope instead of learned positional embeddings", default=False),
    use_mistral_sliding_window=Param(bool, "apply rope instead of learned positional embeddings", default=False),
    init_from=Param(str, "'scratch' or 'best_ckpt' if computing the bounds", default='scratch'),
    model_size=Param(str, "specify size of the model to run", default='small'),
    best_checkpoint_path=Param(str, "path to best checkpoint for bound eval", default=None),
    model_name_or_path=Param(str, "model name or path; could be gpt2 or FacebookAI/roberta-base", default='gpt2'),
)

Section("optimizer", "optimizer details").params(
    learning_rate=Param(float, "adamw optimizer lr", default=6e-4),
    weight_decay=Param(float, "", default=1e-1),
    beta1=Param(float, "", default=0.9),
    beta2=Param(float, "", default=0.95),
    grad_clip=Param(float, "# clip gradients at this value, or disable if == 0.0", default=1.0),
    correct_bias=Param(bool, "", default=False),
    adam_epislon=Param(float, "", default=1e-8),
    no_decay_bias=Param(bool, "", default=True),
)

Section("learning_rate", "learning rate decay settings").params(
    decay_lr=Param(bool, "whether to decay the learning rate", default=True),
    warmup_iters=Param(int, "how many steps to warm up for", default=2000),
    lr_decay_iters=Param(int, "should be ~= max_iters per Chinchilla", default=600000),
    min_lr=Param(float, "minimum learning rate, should be ~= learning_rate/10 per Chinchilla", default=6e-5),
)

Section("sublora", "LoRA and subspace Settings").params(
    use_lora=Param(bool, "true if any LoRA layer is used", default=False),
    lora_alpha=Param(int, "default value", default=32),
    lora_dropout=Param(float, "default value", default=0.1),
    attention_linear_use_lora=Param(bool, "default value", default=False),
    attention_linear_lora_r=Param(int, "", default=1),
    linear_head_lora_r=Param(int, "", default=1),
    linear_head_enable_lora=Param(bool, "", default=False),
    intrinsic_dim=Param(int, "subspace intrinsic dimensionality", default=0),
)

Section("system", "system details").params(
    device=Param(str, "examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks", default='cuda'),
    dtype=Param(str, "'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler",
                default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'),
    compile=Param(bool, "use PyTorch 2.0 to compile the model to be faster", default=True), 
)

Section("bounds", "bound computation details").params(
    use_kmeans=Param(bool, "Using kmeans during the quantization", default=False),
    quant_lr=Param(float, "Learning rate for quantization-aware training", default=5e-5),
    eval_batch_size=Param(int, "Batch size for quantization-aware training and bound evaluation", default=6),
    max_quant_iters=Param(int, "Number of iterations for quantization-aware training", default=0),
    levels=Param(int, "Number of quantization levels, should be odd", default=11),
    bound_samples=Param(int, "number of samples used in the subsampling bounds", default=10000),
    bound_type=Param(str, "'document_level' bounds or 'sequence_level' bounds", default="document_level"),
    sliding_window_size=Param(int, "the length of the sliding window in the evaluation of a doc > 1024 tokens", default=100),
    misc_extra_bits=Param(int, "number of extra bits to be paid for sweeping over multiple hyperparameters", default=5),
)

class SubLoRA():
    @param("data.dataset")
    @param("data.dataset_dir")
    @param("data.block_size")
    @param("data.batch_size")
    @param("data.perturb_word_order_window_size")
    @param("model.init_from")
    @param("model.n_layer")
    @param("model.n_head")
    @param("model.n_embd")
    @param("model.bias")
    @param("model.dropout")
    @param("model.use_mergedlinear")
    @param("model.apply_rope")
    @param("model.use_mistral_sliding_window")
    @param("sublora.use_lora")
    @param("sublora.lora_alpha")
    @param("sublora.lora_dropout")
    @param("sublora.intrinsic_dim")
    @param("sublora.attention_linear_use_lora")
    @param("sublora.attention_linear_lora_r")
    @param("sublora.linear_head_lora_r")
    @param("sublora.linear_head_enable_lora")
    @param("model.model_size")
    @param("model.model_name_or_path")
    @param("system.dtype")
    @param("bounds.eval_batch_size")
    @param("model.best_checkpoint_path")
    def __init__(self, yaml_config, dataset, dataset_dir, block_size, batch_size, perturb_word_order_window_size, init_from, 
                 n_layer, n_head, n_embd, bias, dropout, use_mergedlinear, apply_rope, use_mistral_sliding_window, use_lora, 
                 lora_alpha, lora_dropout, intrinsic_dim, attention_linear_use_lora, attention_linear_lora_r, linear_head_lora_r,
                 linear_head_enable_lora, model_size, model_name_or_path, dtype, eval_batch_size, best_checkpoint_path=None):
        
        ### Change lora config here to train without lora if the rank for both attention and head = 0 
        
        if attention_linear_lora_r == 0:
            attention_linear_use_lora = False
            yaml_config["attention_linear_use_lora"] = attention_linear_use_lora
        if linear_head_lora_r == 0:
            linear_head_enable_lora = False
            yaml_config["linear_head_enable_lora"] = linear_head_enable_lora
        if not (attention_linear_use_lora or linear_head_enable_lora):
            use_lora = False 
            yaml_config["use_lora"] = use_lora
        
        self.yaml_config = yaml_config
        self.block_size = block_size
        self.batch_size = batch_size
        self.perturb_word_order_window_size = perturb_word_order_window_size
        self.intrinsic_dim = intrinsic_dim
        self.use_lora = use_lora
        print("Setting up the ddp.")
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        print("Preparing the common setup.")
        self.prepare_common_setup()
        print("Loading the data.")
        self.data_dir = os.path.join(dataset_dir, dataset)
        self.init_from = init_from
        self.model_name_or_path = model_name_or_path
        self.dataset = dataset
        self.eval_batch_size = eval_batch_size
        self.train_data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
        print("Loading the model.")

        n_layer, n_head, n_embd = get_model_config(model_size)

        yaml_config["n_layer"], yaml_config["n_head"], yaml_config["n_embd"]= n_layer, n_head, n_embd
        
        self.model, self.iter_num, self.best_val_loss, self.model_args, self.nparams  = get_model(n_layer, n_head, n_embd, bias, dropout,
                                                                                    use_mergedlinear, apply_rope, use_mistral_sliding_window, 
                                                                                    use_lora, lora_alpha, lora_dropout, attention_linear_use_lora,
                                                                                    attention_linear_lora_r,linear_head_lora_r, linear_head_enable_lora,
                                                                                    intrinsic_dim, block_size, self.data_dir, self.out_dir, init_from,
                                                                                    self.master_process, self.device, best_checkpoint_path)

        if self.wandb_log and self.master_process:
            wandb.log({"nparams": self.nparams})
        
    @param("optimizer.weight_decay")
    @param("optimizer.learning_rate")
    @param("optimizer.beta1")
    @param("optimizer.beta2")
    @param("optimizer.correct_bias")
    @param("optimizer.adam_epislon")
    @param("optimizer.no_decay_bias")
    @param("system.dtype")
    @param("learning_rate.decay_lr")
    @param("learning_rate.warmup_iters")
    @param("learning_rate.lr_decay_iters")
    @param("learning_rate.min_lr")
    @param("training.eval_interval")
    @param("training.always_save_checkpoint")
    @param("training.eval_only")
    @param("training.gradient_accumulation_steps")
    @param("optimizer.grad_clip")
    @param("training.log_interval")
    @param("training.max_iters")
    @param("system.compile")
    def train(self, weight_decay, learning_rate, beta1, beta2, correct_bias, adam_epislon, no_decay_bias, dtype, decay_lr,
              warmup_iters, lr_decay_iters, min_lr, eval_interval, always_save_checkpoint, eval_only, gradient_accumulation_steps,
              grad_clip, log_interval, max_iters, compile,):
        print("Training begins...")
        iter_num = self.iter_num
        best_val_loss = self.best_val_loss
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), self.device_type,
                                               correct_bias, adam_epislon, no_decay_bias)
        if self.init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if compile and (not self.use_lora) and self.intrinsic_dim == 0:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0
        # wrap model into DDP container
        if self.ddp:
            if self.use_lora:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank], find_unused_parameters=True)
            else:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        if self.use_lora:
            if self.intrinsic_dim == 0:
                lora.mark_only_lora_as_trainable(self.model)
        if self.ddp:
            total_num_params = int(self.model.module.get_num_params())
            total_lora_params = int(self.model.module.get_num_params(only_trainable=True))
            print("number of parameters: %.2fM" % (self.model.module.get_num_params()/1e6,))
            print("number of trainable parameters: %.2f" % (self.model.module.get_num_params(only_trainable=True),))
        else:
            total_num_params = int(self.model.get_num_params())
            total_lora_params = int(self.model.get_num_params(only_trainable=True))
            print("number of parameters: %.2fM" % (self.model.get_num_params()/1e6,))
            print("number of trainable parameters: %.2f" % (self.model.get_num_params(only_trainable=True),))
            
        if self.wandb_log and self.master_process:
            wandb.log({"num_params": total_num_params, "num_lora_params": total_lora_params})

        print("\n# === final trainable parameters === #\n")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n)
        print("\n# === final trainable parameters === #\n")

        torch.manual_seed(1337 + self.seed_offset)
        # training loop
        X, Y, ix = get_batch('train', self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    if losses['val'] < best_val_loss:
                        _best_checkpoint = True
                    else:
                        _best_checkpoint = False

                    best_val_loss = losses['val']
                    if iter_num > 0:
                        # LoRA and conventional training logic
                        if self.use_lora:
                            raw_model_state_dict = raw_model.state_dict()
                            lora_state_dict = lora.lora_state_dict(self.model)
                        else:
                            raw_model_state_dict = raw_model.state_dict()
                            lora_state_dict = None

                        checkpoint = {
                            'raw_model': raw_model_state_dict,
                            'lora_model': lora_state_dict,
                            'optimizer': optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': self.model_args,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        
                        if _best_checkpoint:
                            torch.save(checkpoint, os.path.join(self.out_dir, f'best_ckpt.pt'))
                        else:
                            torch.save(checkpoint, os.path.join(self.out_dir, f'ckpt_{iter_num}.pt'))
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                if self.intrinsic_dim == 0:
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                else:
                    logits, loss = self.model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, ix = get_batch('train', self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break

        if self.ddp:
            destroy_process_group()
        
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    @param("training.eval_iters")
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, ix = get_batch(split, self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
                if self.intrinsic_dim == 0:
                    with self.ctx:
                        _, loss = self.model(X, Y)
                else:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
                
            out[split] = losses.mean()
        self.model.train()
        return out

    @param("system.dtype")
    @param("login.wandb_log")
    @param("login.wandb_project")
    @param("login.wandb_run_name")
    @param("login.create_new_output_dir")
    @param("login.out_dir")
    @param("sublora.intrinsic_dim")
    @param("optimizer.learning_rate")
    @param("sublora.attention_linear_lora_r")
    def prepare_common_setup(self, dtype, wandb_log, wandb_project, wandb_run_name, create_new_output_dir, out_dir,
                             intrinsic_dim, learning_rate, attention_linear_lora_r):
        self.maybe_launch_ddp()
        self.wandb_log = wandb_log
        self.out_dir = out_dir
        wandb_run_name = "id{}_lr{}_r{}".format(intrinsic_dim, learning_rate, attention_linear_lora_r)
        # logging
        if wandb_log and self.master_process:
            wandb.init(project=wandb_project, name=wandb_run_name, config=self.yaml_config)
        # ? creating new output directory
        if create_new_output_dir:
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d')
            formatted_time = now.strftime('%H-%M')
            logging_directory = f'{formatted_date}/{formatted_time}'
            self.out_dir = os.path.join(self.out_dir, wandb_project, wandb_run_name, logging_directory)

        if self.master_process and self.yaml_config["action"] == "train":
            os.makedirs(self.out_dir, exist_ok=True)
        torch.manual_seed(137)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
    @param("training.gradient_accumulation_steps")
    @param("training.backend")
    @param("system.device")
    def maybe_launch_ddp(self, gradient_accumulation_steps, backend, device): 
        self.device = device
        if self.ddp:
            init_process_group(backend=backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert gradient_accumulation_steps % self.ddp_world_size == 0
            gradient_accumulation_steps //= self.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.ddp_rank = 0
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            
            
    @param("bounds.max_quant_iters")
    @param("bounds.use_kmeans")
    @param("bounds.levels")
    @param("bounds.quant_lr")
    @param("bounds.eval_batch_size")
    @param("bounds.bound_samples")
    @param("bounds.bound_type")
    @param("bounds.misc_extra_bits")
    @param("bounds.sliding_window_size")
    @param("model.best_checkpoint_path")
    @param("data.data_size")
    @param("data.eot_token")
    @param("data.vocab_size")
    @param("data.num_docs")
    @param("data.openwebtext_train_eot_indices_file")
    @param("data.empirical_document_length_distribution_file")
    def get_bounds(self, max_quant_iters, use_kmeans, levels, quant_lr, eval_batch_size, bound_samples, bound_type, 
                   misc_extra_bits, sliding_window_size, best_checkpoint_path, data_size, eot_token, vocab_size,
                   num_docs, openwebtext_train_eot_indices_file, empirical_document_length_distribution_file):
        # wrap model into DDP container
        if self.ddp:
            if self.use_lora:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank], find_unused_parameters=True)
            else:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        if self.use_lora:
            if self.intrinsic_dim == 0:
                lora.mark_only_lora_as_trainable(self.model)
        if self.ddp:
            print("number of parameters: %.2fM" % (self.model.module.get_num_params()/1e6,))
            print("number of trainable parameters: %.2f" % (self.model.module.get_num_params(only_trainable=True),))
        else:
            print("number of parameters: %.2fM" % (self.model.get_num_params()/1e6,))
            print("number of trainable parameters: %.2f" % (self.model.get_num_params(only_trainable=True),))

        print("\n# === final trainable parameters === #\n")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n)
        print("\n# === final trainable parameters === #\n")
        
        print("EVALUATING THE MODEL BEFORE QUANTIZATION")

        losses = self.estimate_loss()
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        self.model, prefix_message_len = quantize_model(self.model, self.train_data, self.block_size, self.intrinsic_dim,
                                                        self.device_type, self.device, self.ddp, self.perturb_word_order_window_size,
                                                        eval_batch_size, max_quant_iters, use_kmeans, levels, quant_lr)
    
        print("EVALUATING THE MODEL AFTER QUANTIZATION")
        losses = self.estimate_loss()
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        ################ Saving everything for the bound computation ###################
        raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
        if self.use_lora:
            raw_model_state_dict = raw_model.state_dict()
            lora_state_dict = lora.lora_state_dict(self.model)
        else:
            raw_model_state_dict = raw_model.state_dict()
            lora_state_dict = None

        checkpoint = {
            'raw_model': raw_model_state_dict,
            'lora_model': lora_state_dict,
            'optimizer': None,
            'model_args': self.model_args,
            'iter_num': self.iter_num,
            'best_val_loss': None,
            'config': self.yaml_config,
            'prefix_message_len': prefix_message_len, 
        }
        print(f"saving checkpoint to {best_checkpoint_path}")

        torch.save(checkpoint, os.path.join(best_checkpoint_path, f'quant_ckpt_levels{levels}_iters{max_quant_iters}.pt'))

        # clear file contents
        with open(os.path.join(best_checkpoint_path, f'ix_levels{levels}_iters{max_quant_iters}.txt'), 'w') as file_ix:
            pass 
        with open(os.path.join(best_checkpoint_path, f'top_k_indices_levels{levels}_iters{max_quant_iters}.txt'), 'w') as file_top_k_indices:
            pass 
        with open(os.path.join(best_checkpoint_path, f'selected_prob_scores_levels{levels}_iters{max_quant_iters}.txt'), 'w') as file_selected_prob_scores:
            pass 
        with open(os.path.join(best_checkpoint_path, f'percentile_vec_levels{levels}_iters{max_quant_iters}.txt'), 'w') as file_percentile_vec:
            pass 
        
        alpha_array = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]    
                
        with torch.no_grad():
            self.model.eval()
            curr_iter_i = 0
            metrics_dict = {}

            for k in range(1,10+1):
                metrics_dict[f'top_{k}_acc'] = 0
            metrics_dict[f'top_50_acc'] = 0
            metrics_dict[f'top_100_acc'] = 0
            
            for alpha in alpha_array:
                metrics_dict[f'bpd_alpha_{alpha}'] = 0

            metrics_dict["n_train"] = 0
            metrics_dict["curr_iter_i"] = 0
            
            while bound_samples > metrics_dict["n_train"]:
                print(f"\n +++++ current metrics_dict[n_train] = {metrics_dict['n_train']} +++++ \n")
                        
                if bound_type == "sequence_level":
                    X, Y, ix = sample_nonoverlapping_sequences("train", self.train_data, self.val_data, eval_batch_size, self.block_size,
                                                            self.device_type, self.device, data_size)
                elif bound_type == "document_level":
                    X, Y, ix = sample_single_document("train", self.train_data, self.val_data, eot_token, self.device_type,
                                                    self.device, openwebtext_train_eot_indices_file, empirical_document_length_distribution_file)
                else:
                    raise NotImplemented
                                
                top_k_indices, percentile_vec, selected_prob_scores = compute_bound_scores(self.model, X, Y, self.device,
                                                                                        self.intrinsic_dim, self.block_size,
                                                                                        sliding_window_size, self.ctx)                
                # saving all of ix & top-k & percentile_vec
                with open(os.path.join(best_checkpoint_path, f'ix_levels{levels}_iters{max_quant_iters}.txt'), 'a') as file_ix:
                    file_ix.write(str(ix.tolist()) + '\n') 
                with open(os.path.join(best_checkpoint_path, f'top_k_indices_levels{levels}_iters{max_quant_iters}.txt'), 'a') as file_top_k_indices:
                    file_top_k_indices.write(str(top_k_indices.tolist()) + '\n')  
                with open(os.path.join(best_checkpoint_path, f'selected_prob_scores_levels{levels}_iters{max_quant_iters}.txt'), 'a') as file_selected_prob_scores:
                    file_selected_prob_scores.write(str(selected_prob_scores.tolist()) + '\n')  
                with open(os.path.join(best_checkpoint_path, f'percentile_vec_levels{levels}_iters{max_quant_iters}.txt'), 'a') as file_percentile_vec:
                    file_percentile_vec.write(str(percentile_vec.tolist()) + '\n')  
                    
                metrics_dict = compute_bound_metrics(metrics_dict, top_k_indices, selected_prob_scores, alpha_array,
                                                    bound_type, eval_batch_size, vocab_size, len_x=X.shape[1])
                with open(os.path.join(best_checkpoint_path, f'metrics_levels{levels}_iters{max_quant_iters}.yml'), 'w') as f:
                                yaml.safe_dump(metrics_dict, f, indent=2)

                if self.wandb_log:
                    wandb.log(metrics_dict)

                if curr_iter_i % 100 == 0:
                    print("\n".join("{}\t{}".format(k, v) for k, v in metrics_dict.items()))

                curr_iter_i += 1                
                
        # prefix_message_len = torch.load(os.path.join(best_checkpoint_path, f'quant_ckpt_levels{levels}_iters{max_quant_iters}.pt'))['prefix_message_len']
        sample_size = metrics_dict["n_train"]

        bounds_dict = {}
        bounds_dict["prefix_message_len"] = float(prefix_message_len)
        
        best_bpd_bound = np.inf
                
        print(f"metrics_dict={metrics_dict}")
        print(f"bounds_dict={bounds_dict}")

        total_sample_size = data_size if bound_type == "sequence_level" else num_docs 

        for k in metrics_dict.keys():
            if k != "n_train" and k != "curr_iter_i":
                if "acc" in k:
                    train_error = 1. - metrics_dict[k] 
                    divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
                    bounds_dict["acc_divergence"] = float(divergence)
                    bounds_dict[f"bound_{k}"] = float(llm_subsampling_bound(train_error=train_error,
                                                        div=divergence,
                                                        data_size=total_sample_size,
                                                        sample_size=sample_size,
                                                        delta=1.))
                else:
                    misc_extra_bits += np.ceil(len(alpha_array))
                    divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
                    bounds_dict["bpd_divergence"] = float(divergence)
                    alpha = float(k.replace("bpd_alpha_", ""))
                    delta = np.log2(1 + (1 - alpha) * vocab_size / alpha)
                    train_error = metrics_dict[k]
                    bounds_dict[f"bound_{k}"] = float(llm_subsampling_bound(train_error=train_error,
                                                                            div=divergence,
                                                                            data_size=total_sample_size,
                                                                            sample_size=sample_size,
                                                                            delta=delta))
                    
                    if best_bpd_bound > bounds_dict[f"bound_{k}"]:
                        best_bpd_bound = bounds_dict[f"bound_{k}"]
                                
        bounds_dict["best_bpd_bound"] = best_bpd_bound
        
        print("\n".join("{}\t{}".format(k, v) for k, v in bounds_dict.items()))

        if self.wandb_log:
            wandb.log(bounds_dict)
            
        with open(os.path.join(best_checkpoint_path, f'bounds_levels{levels}_iters{max_quant_iters}.yml'), 'w') as f:
                                yaml.safe_dump(bounds_dict, f, indent=2)