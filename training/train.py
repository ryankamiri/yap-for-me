import argparse
import os
import signal
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from config import Config
from dataloader import create_datasets


checkpoint_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM signal from SLURM before timeout."""
    global checkpoint_requested
    print("\nReceived SIGTERM - will save checkpoint and exit gracefully...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    checkpoint_requested = True


signal.signal(signal.SIGTERM, signal_handler)


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def log_gpu_memory(stage=""):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        print(f"GPU Memory {stage}:")
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


def find_latest_checkpoint(job_id):
    """Find the latest checkpoint in a job's output directory."""
    job_dir = Path(f"out/{job_id}")
    if not job_dir.exists():
        return None
    
    checkpoints = []
    for item in job_dir.iterdir():
        if item.is_dir() and (item / "training_state.pt").exists():
            state = torch.load(item / "training_state.pt", map_location='cpu', weights_only=False)
            checkpoints.append((item, state.get('step', 0)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]


def load_checkpoint(checkpoint_dir, device, model_kwargs=None):
    """Load model and training state from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"Loading model from checkpoint: {checkpoint_dir}")
    if model_kwargs is None:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs)
    model.to(device)
    
    state_path = checkpoint_dir / "training_state.pt"
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    
    resume_epoch = state['epoch']
    resume_step = state['step']
    resume_batch_idx = state.get('batch_idx', 0)
    best_val_loss = state.get('best_val_loss', float('inf'))
    wandb_run_id = state.get('wandb_run_id', None)
    optimizer_state = state.get('optimizer_state_dict', None)
    
    print(f"Resuming from epoch {resume_epoch}, step {resume_step}, batch {resume_batch_idx}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")
    
    return model, resume_epoch, resume_step, resume_batch_idx, best_val_loss, wandb_run_id, optimizer_state


def save_checkpoint(model, tokenizer, optimizer, epoch, step, batch_idx, best_val_loss, wandb_run_id, output_dir, name):
    """Save model checkpoint with all state needed for resume."""
    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    checkpoint_state = {
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx,
        'best_val_loss': best_val_loss,
        'wandb_run_id': wandb_run_id,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint_state, checkpoint_dir / "training_state.pt")
    print(f"Checkpoint saved: {checkpoint_dir}")


def main():
    global checkpoint_requested
    
    parser = argparse.ArgumentParser(description="Train YapForMe model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Job ID to resume from (e.g., 3089731). Will find latest checkpoint automatically."
    )
    args = parser.parse_args()
    
    config = Config(args.config)
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    print(f"SLURM Job ID: {slurm_job_id}")
    
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(args.resume)
        if resume_checkpoint is None:
            print(f"Error: No checkpoint found in out/{args.resume}/")
            sys.exit(1)
        print(f"Found checkpoint to resume: {resume_checkpoint}")
    
    model_name = config.get("model.name")
    max_length = config.get("model.max_length")
    tokenized_examples_path = config.get("dataset.tokenized_examples_path")
    padding_side = config.get("dataset.padding_side")
    random_seed = config.get("dataset.random_seed")

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(random_seed)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using torch device: {device}")
    
    # Create generator for reproducible DataLoader shuffling
    # This ensures shuffling is deterministic across runs but different each epoch
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **config.get("model.tokenizer_kwargs", {})
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    
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
            print(f"Converting dtype '{dtype_str}' to torch.{dtype_str}")
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    log_gpu_memory("Before model load")
    
    if resume_checkpoint is None:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        model.to(device)
    
    gradient_checkpointing = config.get("training.gradient_checkpointing", False)
    
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
    
    batch_size = config.get("training.batch_size")
    gradient_accumulation_steps = config.get("training.gradient_accumulation_steps")
    num_epochs = config.get("training.epochs")
    learning_rate = float(config.get("training.learning_rate"))
    eval_steps = config.get("training.eval_steps")
    save_steps = config.get("training.save_steps")
    dataloader_workers = config.get("training.dataloader_workers")
    periodic_checkpoint_interval = config.get("training.periodic_checkpoint_interval")
    
    # Use SLURM job output directory: out/{job_id}/
    output_dir = f"out/{slurm_job_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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
    
    if resume_checkpoint is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        start_epoch = 0
        global_step = 0
        resume_batch_idx = 0
        best_val_loss = float('inf')
        wandb_run_id = None
    else:
        model, start_epoch, global_step, resume_batch_idx, best_val_loss, wandb_run_id, optimizer_state = load_checkpoint(
            resume_checkpoint, device, model_kwargs
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state from checkpoint")
        
        # Clear memory fragmentation after loading checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_epoch = start_epoch - 1
    
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        else:
            print("Warning: Gradient checkpointing requested but not supported by this model")
    else:
        print("Gradient checkpointing disabled (config: false)")
    
    log_gpu_memory("After model load")
    
    log_gpu_memory("After optimizer creation")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print()
    print(f"Training Configuration:")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {output_dir}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
        print(f"Starting at epoch {start_epoch + 1}, step {global_step}, batch {resume_batch_idx}")
    
    if wandb_run_id:
        wandb.init(
            project="yap-for-me",
            id=wandb_run_id,
            resume="must",
        )
        print(f"Resumed wandb run: {wandb_run_id}")
    else:
        wandb.init(
            project="yap-for-me",
            name=f"{model_name.split('/')[-1]}-{slurm_job_id}",
            config={
                "model_name": model_name,
                "max_length": max_length,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "random_seed": random_seed,
                "slurm_job_id": slurm_job_id,
            }
        )
        wandb_run_id = wandb.run.id
        print(f"Started new wandb run: {wandb_run_id}")
    
    last_checkpoint_time = time.time() if periodic_checkpoint_interval > 0 else None
    
    # Clear memory before training starts (especially important when resuming)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_gpu_memory("Before training loop starts")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_num = epoch + 1
        print()
        print(f"{'='*60}")
        print(f"Epoch {epoch_num}/{num_epochs}")
        print(f"{'='*60}")
        
        model.train()
        total_train_loss = 0
        num_batches = len(train_loader)
        
        optimizer.zero_grad()
        
        skip_batches = resume_batch_idx if epoch == start_epoch else 0
        if skip_batches > 0:
            print(f"Skipping first {skip_batches} batches (already processed)...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue
            
            if checkpoint_requested:
                print(f"\nSaving checkpoint before timeout at epoch {epoch_num}, step {global_step}, batch {batch_idx}...")
                save_checkpoint(
                    model, tokenizer, optimizer, epoch_num, global_step, batch_idx,
                    best_val_loss, wandb_run_id, output_dir, "timeout-checkpoint"
                )
                print("Checkpoint saved. Exiting gracefully.")
                wandb.finish()
                sys.exit(0)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # (B, L)
            
            if batch_idx == skip_batches:
                log_gpu_memory("Before first forward pass")
            
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
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if batch_idx == skip_batches:
                log_gpu_memory("After first backward pass")
            
            # Only step optimizer every gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if batch_idx == skip_batches:
                    log_gpu_memory("After first optimizer step")
            
            # Track unscaled loss for logging
            total_train_loss += loss.item() * gradient_accumulation_steps
            
            # Log loss at each accumulation step (scaled back up for display)
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/epoch": epoch_num,
                }, step=global_step)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, GPU Memory: {format_bytes(allocated)}")
                else:
                    print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                
                # Periodic checkpoint save as backup
                # Signals can be blocked during CUDA operations, so this ensures we don't lose progress
                if periodic_checkpoint_interval > 0 and last_checkpoint_time is not None:
                    current_time = time.time()
                    if current_time - last_checkpoint_time >= periodic_checkpoint_interval:
                        print(f"\nPeriodic checkpoint save (every {periodic_checkpoint_interval // 60} minutes)...")
                        save_checkpoint(
                            model, tokenizer, optimizer, epoch_num, global_step, batch_idx + 1,
                            best_val_loss, wandb_run_id, output_dir, "periodic-checkpoint"
                        )
                        last_checkpoint_time = current_time
                        print("Periodic checkpoint saved.")
                
                if checkpoint_requested:
                    print(f"\nSaving checkpoint before timeout at epoch {epoch_num}, step {global_step}, batch {batch_idx + 1}...")
                    save_checkpoint(
                        model, tokenizer, optimizer, epoch_num, global_step, batch_idx + 1,
                        best_val_loss, wandb_run_id, output_dir, "timeout-checkpoint"
                    )
                    print("Checkpoint saved. Exiting gracefully.")
                    wandb.finish()
                    sys.exit(0)
            
            if global_step > 0 and global_step % eval_steps == 0:
                val_loss = evaluate_model(model, val_loader, criterion, device)
                print(f"Step {global_step}: Validation Loss: {val_loss:.4f}")
                
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/step": global_step,
                }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, tokenizer, optimizer, epoch_num, global_step, batch_idx + 1,
                        best_val_loss, wandb_run_id, output_dir, "best"
                    )
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
                    wandb.log({"eval/best_loss": best_val_loss}, step=global_step)
                
                model.train()
            
            if global_step > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    model, tokenizer, optimizer, epoch_num, global_step, batch_idx + 1,
                    best_val_loss, wandb_run_id, output_dir, f"checkpoint-{global_step}"
                )
        
        batches_processed = num_batches - skip_batches
        if batches_processed > 0:
            avg_train_loss = total_train_loss / batches_processed
        else:
            avg_train_loss = 0
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss": val_loss,
            "epoch/epoch": epoch_num,
        }, step=global_step)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer, optimizer, epoch_num, global_step, 0,
                best_val_loss, wandb_run_id, output_dir, "best"
            )
            print(f"Saved best model (val_loss: {val_loss:.4f})")
            wandb.log({"epoch/best_val_loss": best_val_loss}, step=global_step)
        
        save_checkpoint(
            model, tokenizer, optimizer, epoch_num, global_step, 0,
            best_val_loss, wandb_run_id, output_dir, f"epoch-{epoch_num}"
        )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    
    save_checkpoint(
        model, tokenizer, optimizer, epoch_num, global_step, 0,
        best_val_loss, wandb_run_id, output_dir, "final"
    )
    
    wandb.log({"final/best_val_loss": best_val_loss})
    wandb.finish()
    
    print(f"\nFinal model saved to {output_dir}")


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
    
    model.train()
    return total_val_loss / len(val_loader)

# Target loss goals for production-ready model:
# - Validation loss: 1.0-1.3 (excellent), 1.3-1.5 (good), <1.0 (SOTA)
# - Test loss: Should be within 0.1-0.2 of validation loss (good generalization)
# - Training/val gap: <0.2 indicates good generalization, >0.3 suggests overfitting

if __name__ == "__main__":
    main()
