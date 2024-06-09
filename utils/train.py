from dataclasses import dataclass
from pathlib import Path
import math

from safetensors.torch import save_model, load_model
import torch

import wandb


@dataclass
class TrainConfig():
    exp_name: str = 'default'

    batch_size: int = 256
    grad_accum: int = 1

    p_augs: float = 0.0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    max_steps: int = 100_000
    eval_interval: int = 1_000
    
    use_scheduler: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 50_000
    
    num_workers: int = 3
    pin_memory: bool = True
    
    grad_clip: float = 1.0
    mixed_precision: bool = False

    visualize_predictions: bool = False

def init_lr_scheduler(config):
    learning_rate = config.learning_rate
    warmup_iters = config.warmup_iters
    lr_decay_iters = config.lr_decay_iters
    min_lr = learning_rate / 10
    constant_lr = not config.use_scheduler

    def get_lr(it):
        if constant_lr: 
            return learning_rate
        # 1) linear warmup for warmup_iters steps.
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
    
    print('Completed initialization of scheduler')
    return get_lr

def prepare_data_loaders(train_dataset, val_dataset, config):
    """Prepare the training and validation data loaders."""
    batch_size = config.batch_size // config.grad_accum
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    return train_loader, val_loader

def run_train_model(model, datasets, config, device='cuda'):

    SAVE_FOLDER = Path('logs') / config.exp_name
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = datasets
    train_loader, val_loader = prepare_data_loaders(train_dataset, val_dataset, config)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.learning_rate, 
                                  weight_decay=config.weight_decay)
    scheduler = init_lr_scheduler(config)
    
    overall_step = 0
    best_val_loss = float('inf')

    # Move model to the specified device
    model.to(device)

    while True:
        for batch in train_loader:
            lr = scheduler(overall_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad(set_to_none=True)
            
            inputs, labels = batch
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            loss, _ = model(inputs, labels)
            loss.backward()
            
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            overall_step += 1
            wandb.log({'train/loss': loss.item(), 'lr': lr}, step=overall_step)
            print('*', end='')

            if (overall_step % config.eval_interval) == 0:
                model.eval()
                val_loss_list = []
                for batch in val_loader:
                    inputs, labels = batch
                    # Move data to the specified device
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        val_loss, _ = model(inputs, labels)
                    val_loss_list.append(val_loss)
                
                mean_val_loss = torch.stack(val_loss_list).mean()

                print('\n')
                print(f"overall_steps {overall_step}: {loss.item()}")
                print(f"val loss: {mean_val_loss}")
                wandb.log({'val/loss': mean_val_loss}, step=overall_step)
            
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    save_path = SAVE_FOLDER / f"step_{overall_step}_loss_{mean_val_loss:.4f}.safetensors"
                    save_model(model, save_path)
                    print('saved model: ', save_path.name)
                print('\n')
                model.train()
            
            if overall_step > config.max_steps:
                print('Complete training')
                break