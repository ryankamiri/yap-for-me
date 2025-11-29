import argparse
import multiprocessing as mp
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import Config
from dataloader import build_training_examples_from_conversations, load_conversations


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of worker processes (default: CPU count)"
    )
    args = parser.parse_args()
    
    config = Config(args.config)
    
    model_name = config.get("model.name")
    max_length = config.get("model.max_length")
    conversations_path = config.get("dataset.path")
    tokenized_examples_path = config.get("dataset.tokenized_examples_path")
    
    num_workers = args.num_workers if args.num_workers else mp.cpu_count()
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **config.get("model.tokenizer_kwargs", {})
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading conversations from: {conversations_path}")
    conversations = load_conversations(conversations_path)
    print(f"Loaded {len(conversations)} conversations")
    
    print(f"Building training examples with {num_workers} workers (this may take a while)...")
    
    chunk_size = 10
    conversation_chunks = []
    for i in range(0, len(conversations), chunk_size):
        chunk = [(i+j, conv) for j, conv in enumerate(conversations[i:i+chunk_size])]
        conversation_chunks.append(chunk)
    
    # Process conversations in parallel
    all_examples = []
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(model_name, config.get("model.tokenizer_kwargs", {}), max_length)) as pool:
        results = pool.imap(process_conversation_batch, conversation_chunks)
        
        for chunk_idx, conv_examples in enumerate(results):
            all_examples.extend(conv_examples)
            conversations_processed = min((chunk_idx + 1) * chunk_size, len(conversations))
            if conversations_processed % 100 == 0 or chunk_idx == len(conversation_chunks) - 1:
                print(f"  Processed {conversations_processed}/{len(conversations)} conversations... ({len(all_examples)} examples so far)")
    
    print(f"Created {len(all_examples)} training examples")
    
    output_path = Path(tokenized_examples_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving tokenized examples to: {output_path}")
    torch.save(all_examples, output_path)
    
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"Saved {len(all_examples):,} examples to {output_path}")
    print(f"File size: {file_size_gb:.2f} GB")


# Global tokenizer for worker processes
_worker_tokenizer = None
_worker_max_length = None


def init_worker(model_name, tokenizer_kwargs, max_length):
    """Initialize worker process with tokenizer and parameters."""
    global _worker_tokenizer, _worker_max_length
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if _worker_tokenizer.pad_token is None:
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
    _worker_max_length = max_length


def process_conversation_batch(chunk):
    """Process a batch of conversations in a worker process."""
    all_examples = []
    for _, conversation in chunk:
        # Build examples for this conversation
        conv_examples = build_training_examples_from_conversations(
            [conversation],
            _worker_tokenizer,
            max_length=_worker_max_length
        )
        
        # Add conversation metadata to each example
        conversation_id = conversation['chat_session']
        for example in conv_examples:
            example['conversation_id'] = conversation_id
        
        all_examples.extend(conv_examples)
    
    return all_examples


if __name__ == "__main__":
    main()

# Extract command:
# rsync -avz --progress amiri.ry@login.explorer.northeastern.edu:/projects/llpr/amiri.ry/projects/yap-for-me/training/data/tokenized_examples.pt /Users/ramiri/dev/projects/YapForMe/training/data/

