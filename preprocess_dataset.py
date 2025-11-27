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
    outgoing_speaker_name = config.get("dataset.outgoing_speaker_name")
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
    
    # Process conversations in parallel
    all_examples = []
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(model_name, config.get("model.tokenizer_kwargs", {}), max_length, outgoing_speaker_name)) as pool:
        results = pool.imap(process_conversation_worker, enumerate(conversations))
        
        for idx, conv_examples in enumerate(results):
            all_examples.extend(conv_examples)
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(conversations)} conversations... ({len(all_examples)} examples so far)")
    
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
_worker_outgoing_speaker_name = None


def init_worker(model_name, tokenizer_kwargs, max_length, outgoing_speaker_name):
    """Initialize worker process with tokenizer and parameters."""
    global _worker_tokenizer, _worker_max_length, _worker_outgoing_speaker_name
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if _worker_tokenizer.pad_token is None:
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
    _worker_max_length = max_length
    _worker_outgoing_speaker_name = outgoing_speaker_name


def process_conversation_worker(args):
    """Process a single conversation in a worker process."""
    _, conversation = args
    
    # Build examples for this conversation
    conv_examples = build_training_examples_from_conversations(
        [conversation],
        _worker_tokenizer,
        max_length=_worker_max_length,
        outgoing_speaker_name=_worker_outgoing_speaker_name
    )
    
    # Add conversation metadata to each example
    conversation_id = conversation['chat_session']
    for example in conv_examples:
        example['conversation_id'] = conversation_id
    
    return conv_examples


if __name__ == "__main__":
    main()

