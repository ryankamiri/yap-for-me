import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from config import Config
from dataloader import create_datasets


def main():
    parser = argparse.ArgumentParser(description="Train YapForMe model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    config = Config(args.config)
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    print(f"SLURM Job ID: {slurm_job_id}")
    
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **config.get("model.model_kwargs", {})
    )
    model.to(device)
    
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
    num_epochs = config.get("training.epochs")
    learning_rate = float(config.get("training.learning_rate"))
    eval_steps = config.get("training.eval_steps")
    save_steps = config.get("training.save_steps")
    
    # Use SLURM job output directory: out/{job_id}/
    output_dir = f"out/{slurm_job_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print()
    print(f"Training Configuration:")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {output_dir}")
    
    wandb.init(
        project="yap-for-me",
        name=f"{model_name.split('/')[-1]}-{slurm_job_id}",
        config={
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
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
        epoch = epoch + 1
        print()
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        model.train()
        total_train_loss = 0
        num_batches = len(train_loader)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # (B, L)
            
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
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            total_train_loss += loss.item()
            
            wandb.log({
                "train/loss": loss.item(),
                "train/step": global_step,
                "train/epoch": epoch,
            }, step=global_step)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
            
            if global_step % eval_steps == 0:
                val_loss = evaluate_model(model, val_loader, criterion, device)
                print(f"Step {global_step}: Validation Loss: {val_loss:.4f}")
                
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/step": global_step,
                }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, tokenizer, optimizer, epoch, global_step, val_loss, output_dir, "best")
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
                    wandb.log({"eval/best_loss": best_val_loss}, step=global_step)
            
            if global_step % save_steps == 0:
                save_checkpoint(model, tokenizer, optimizer, epoch, global_step, None, output_dir, f"checkpoint-{global_step}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss": val_loss,
            "epoch/epoch": epoch,
        }, step=global_step)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, optimizer, epoch, global_step, val_loss, output_dir, "best")
            print(f"Saved best model (val_loss: {val_loss:.4f})")
            wandb.log({"epoch/best_val_loss": best_val_loss}, step=global_step)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    
    save_checkpoint(model, tokenizer, optimizer, epoch, global_step, best_val_loss, output_dir, "final")
    
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


if __name__ == "__main__":
    main()
