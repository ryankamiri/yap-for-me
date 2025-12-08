import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from config import Config
from dataloader import create_datasets


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def log_gpu_memory(stage="", device=None):
    """Log GPU memory usage."""
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    if device is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        
        print(f"GPU Memory {stage} (GPU {device}):")
        print(f"Allocated: {format_bytes(allocated)} ({allocated / total * 100:.1f}%)")
        print(f"Reserved: {format_bytes(reserved)} ({reserved / total * 100:.1f}%)")
        print(f"Total: {format_bytes(total)}")
        print(f"Free: {format_bytes(total - reserved)}")
        print()
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved
        }
    return None


def setup_distributed():
    """
    Initialize distributed training if running with torchrun.
    Returns: (rank, local_rank, world_size, device, is_main_process)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        is_main_process = (rank == 0)
        
        return rank, local_rank, world_size, device, is_main_process
    
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        
        return rank, local_rank, world_size, device, is_main_process


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train YapForMe model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    rank, local_rank, world_size, device, is_main_process = setup_distributed()
    
    config = Config(args.config)
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    
    if is_main_process:
        print(f"SLURM Job ID: {slurm_job_id}")
        print(f"Training on {world_size} GPU(s)")
        if world_size > 1:
            print(f"Rank {rank}/{world_size}, Local Rank: {local_rank}, Device: {device}")
    
    model_name = config.get("model.name")
    max_length = config.get("model.max_length")
    tokenized_examples_path = config.get("dataset.tokenized_examples_path")
    padding_side = config.get("dataset.padding_side")
    random_seed = config.get("dataset.random_seed")

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Create generator for reproducible DataLoader shuffling
    # This ensures shuffling is deterministic across runs but different each epoch
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    if is_main_process:
        print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **config.get("model.tokenizer_kwargs", {})
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_main_process:
        print(f"Loading model: {model_name}")
        log_gpu_memory("Before model load", local_rank)
    
    # Convert dtype string to torch dtype if needed
    model_kwargs = config.get("model.model_kwargs", {}).copy()
    if "dtype" in model_kwargs:
        dtype_str = model_kwargs.pop("dtype")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_str in dtype_map:
            model_kwargs["dtype"] = dtype_map[dtype_str]
            if is_main_process:
                print(f"Converting dtype '{dtype_str}' to torch.{dtype_str}")
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    model.to(device)
    
    gradient_checkpointing = config.get("training.gradient_checkpointing", False)
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_main_process:
                print("Gradient checkpointing enabled")
        else:
            if is_main_process:
                print("Warning: Gradient checkpointing requested but not supported by this model")
    else:
        if is_main_process:
            print("Gradient checkpointing disabled")
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process:
            print("Model wrapped with DistributedDataParallel")
    
    if is_main_process:
        log_gpu_memory("After model load", local_rank)
    
    if is_main_process:
        print(f"Loading pre-tokenized examples from: {tokenized_examples_path}")
    
    train_dataset, val_dataset, _ = create_datasets(
        tokenized_examples_path=tokenized_examples_path,
        tokenizer=tokenizer,
        max_length=max_length,
        train_split=config.get("dataset.train_split"),
        val_split=config.get("dataset.val_split"),
        test_split=config.get("dataset.test_split"),
        random_seed=random_seed,
        padding_side=padding_side
    )
    
    if is_main_process:
        print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    batch_size = config.get("training.batch_size")
    gradient_accumulation_steps = config.get("training.gradient_accumulation_steps", 1)
    num_epochs = config.get("training.epochs")
    learning_rate = float(config.get("training.learning_rate"))
    eval_steps = config.get("training.eval_steps")
    save_steps = config.get("training.save_steps")
    dataloader_workers = config.get("training.dataloader_workers", 4)
    
    # Use SLURM job output directory: out/{job_id}/
    output_dir = f"out/{slurm_job_id}"
    if is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        generator=generator if world_size == 1 else None,
        num_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )
    
    if is_main_process:
        log_gpu_memory("After optimizer creation", local_rank)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    
    if is_main_process:
        print()
        print(f"Training Configuration:")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size (per GPU): {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Number of GPUs: {world_size}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Output directory: {output_dir}")
    
    if is_main_process:
        wandb.init(
            project="yap-for-me",
            name=f"{model_name.split('/')[-1]}-{slurm_job_id}",
            config={
                "model_name": model_name,
                "max_length": max_length,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_gpus": world_size,
                "effective_batch_size": effective_batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "random_seed": random_seed,
                "slurm_job_id": slurm_job_id,
            }
        )
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print()
            print(f"{'='*60}")
            print(f"Epoch {epoch_num}/{num_epochs}")
            print(f"{'='*60}")
        
        model.train()
        total_train_loss = 0
        num_batches = len(train_loader)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # (B, L)
            
            if batch_idx == 0 and is_main_process:
                log_gpu_memory("Before first forward pass", local_rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # (B, L, V)
            
            # Shift for next-token prediction: align logits[i] with labels[i+1]
            # logits shape: [B, L, V] where B=batch, L=seq_len, V=vocab_size
            # labels shape: [B, L]
            shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
            shift_labels = labels[..., 1:].contiguous()       # [B, L-1]
            
            # Reshape for loss calculation: flatten batch and sequence dimensions
            # shift_logits: [B, L-1, V] -> [B*(L-1), V]
            # shift_labels: [B, L-1] -> [B*(L-1)]
            B, L_minus_1, V = shift_logits.shape
            flat_logits = shift_logits.view(B * L_minus_1, V)
            flat_labels = shift_labels.view(B * L_minus_1)
            
            loss = criterion(flat_logits, flat_labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if batch_idx == 0 and is_main_process:
                log_gpu_memory("After first backward pass", local_rank)
            
            # Only step optimizer every gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if batch_idx == 0 and is_main_process:
                    log_gpu_memory("After first optimizer step", local_rank)
            
            # Track unscaled loss for logging
            total_train_loss += loss.item() * gradient_accumulation_steps
            
            # Log loss at each accumulation step (scaled back up for display)
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if is_main_process:
                    wandb.log({
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/epoch": epoch_num,
                    }, step=global_step)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                if is_main_process:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(device)
                        print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, GPU Memory: {format_bytes(allocated)}")
                    else:
                        print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            if is_main_process and global_step > 0 and global_step % eval_steps == 0:
                model_to_eval = model.module if world_size > 1 else model
                val_loss = evaluate_model(model_to_eval, val_loader, criterion, device)
                print(f"Step {global_step}: Validation Loss: {val_loss:.4f}")
                
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/step": global_step,
                }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model_to_eval, tokenizer, optimizer, epoch_num, global_step, val_loss, output_dir, "best")
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
                    wandb.log({"eval/best_loss": best_val_loss}, step=global_step)
                
                model.train()
            
            if is_main_process and global_step > 0 and global_step % save_steps == 0:
                model_to_save = model.module if world_size > 1 else model
                save_checkpoint(model_to_save, tokenizer, optimizer, epoch_num, global_step, None, output_dir, f"checkpoint-{global_step}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        if is_main_process:
            print(f"\nAverage training loss: {avg_train_loss:.4f}")
            
            model_to_eval = model.module if world_size > 1 else model
            val_loss = evaluate_model(model_to_eval, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")
            
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/epoch": epoch_num,
            }, step=global_step)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model_to_eval, tokenizer, optimizer, epoch_num, global_step, val_loss, output_dir, "best")
                print(f"Saved best model (val_loss: {val_loss:.4f})")
                wandb.log({"epoch/best_val_loss": best_val_loss}, step=global_step)
            
            model.train()
    
    if is_main_process:
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        
        model_to_save = model.module if world_size > 1 else model
        save_checkpoint(model_to_save, tokenizer, optimizer, epoch_num, global_step, best_val_loss, output_dir, "final")
        
        wandb.log({"final/best_val_loss": best_val_loss})
        wandb.finish()
        
        print(f"\nFinal model saved to {output_dir}")
    
    cleanup_distributed()


def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next-token prediction: align logits[i] with labels[i+1]
            # logits shape: [B, L, V] where B=batch, L=seq_len, V=vocab_size
            # labels shape: [B, L]
            shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
            shift_labels = labels[..., 1:].contiguous()       # [B, L-1]
            
            # Reshape for loss calculation: flatten batch and sequence dimensions
            # shift_logits: [B, L-1, V] -> [B*(L-1), V]
            # shift_labels: [B, L-1] -> [B*(L-1)]
            B, L_minus_1, V = shift_logits.shape
            flat_logits = shift_logits.view(B * L_minus_1, V)
            flat_labels = shift_labels.view(B * L_minus_1)
            
            loss = criterion(flat_logits, flat_labels)
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)


def save_checkpoint(model, tokenizer, optimizer, epoch, step, val_loss, output_dir, name):
    """Save model checkpoint."""
    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    checkpoint_state = {
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if val_loss is not None:
        checkpoint_state['val_loss'] = val_loss
    
    torch.save(checkpoint_state, checkpoint_dir / "training_state.pt")


# Target loss goals for production-ready model:
# - Validation loss: 1.0-1.3 (excellent), 1.3-1.5 (good), <1.0 (SOTA)
# - Test loss: Should be within 0.1-0.2 of validation loss (good generalization)
# - Training/val gap: <0.2 indicates good generalization, >0.3 suggests overfitting


if __name__ == "__main__":
    main()
