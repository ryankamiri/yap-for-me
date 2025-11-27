import json
from typing import List, Dict, Tuple
import random

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def format_message(timestamp: str, speaker: str, text: str, replying_to: str = None) -> str:
    """Format a message into the standard format: [timestamp] Speaker: text
    If replying_to is provided, it's included before the text.
    """
    if replying_to:
        return f"[{timestamp}] {speaker}: {replying_to} {text}"
    return f"[{timestamp}] {speaker}: {text}"


def has_incoming_context(messages: List[Dict], current_idx: int) -> bool:
    """Check if there's at least one Incoming message before the current index."""
    for i in range(current_idx):
        if messages[i]['type'] == 'Incoming':
            return True
    return False


def build_training_examples_from_conversations(
    conversations: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    outgoing_speaker_name: str = "Ryan Amiri"
) -> List[Dict]:
    """Build training examples from conversations.
    
    For each Outgoing message in each conversation:
    - Check if there's at least one Incoming message before it
    - If yes, create a training example with context + target
    - Context tokens get -100 labels, target tokens get normal labels
    
    Args:
        conversations: List of conversation dicts from JSON
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        outgoing_speaker_name: Name to use for outgoing messages
        
    Returns:
        List of training examples with 'input_ids' and 'labels'
    """
    examples = []
    
    for _, conversation in enumerate(conversations):
        messages = conversation['messages']
        
        for msg_idx, target_message in enumerate(messages):
            if target_message['type'] != 'Outgoing':
                continue
            
            if not has_incoming_context(messages, msg_idx):
                continue
            
            context_messages = messages[:msg_idx]
            
            context_texts = [
                format_message(
                    msg['timestamp'],
                    msg['speaker'],
                    msg['text'],
                    msg['replying_to']
                )
                for msg in context_messages
            ]
            target_text = format_message(
                target_message['timestamp'],
                target_message['speaker'],
                target_message['text'],
                target_message['replying_to']
            )
            
            context_text = '\n'.join(context_texts)
            
            context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
            
            # Truncate context tokens if they are too long
            if len(context_tokens) > max_length - len(target_tokens):
                available_context = max_length - len(target_tokens)
                if available_context <= 0:
                    continue
                context_tokens = context_tokens[-available_context:]
            
            # Handle case where context + target exceeds max_length
            # This can happen if the target message itself is very long
            # We truncate the target to fit, but skip if there's no room for any target tokens
            total_length = len(context_tokens) + len(target_tokens)
            if total_length > max_length:
                available_for_target = max_length - len(context_tokens)
                if available_for_target <= 0:
                    continue
                target_tokens = target_tokens[:available_for_target]
            
            input_ids = context_tokens + target_tokens
            labels = [-100] * len(context_tokens) + target_tokens
            
            examples.append({
                'input_ids': input_ids,
                'labels': labels,
                'context_length': len(context_tokens),
                'target_length': len(target_tokens)
            })
    
    return examples


def split_conversations(
    conversations: List[Dict],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split conversations into train/val/test sets.
    
    Args:
        conversations: List of conversation dicts
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for test
        random_seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_conversations, val_conversations, test_conversations)
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    # Shuffle conversations (not messages within conversations)
    # Conversations can be shuffled because they're independent
    # Messages within conversations must maintain chronological order
    random.seed(random_seed)
    shuffled = conversations.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_convos = shuffled[:train_end]
    val_convos = shuffled[train_end:val_end]
    test_convos = shuffled[val_end:]
    
    return train_convos, val_convos, test_convos


class ConversationDataset(Dataset):
    """PyTorch Dataset for conversation training examples."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        padding_side: str = "left"
    ):
        """Initialize dataset.
        
        Args:
            examples: List of training examples with 'input_ids' and 'labels'
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for padding
            padding_side: "left" or "right" - where to pad sequences
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side
        
        if padding_side not in ["left", "right"]:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        input_ids = example['input_ids']
        labels = example['labels']
        
        padding_length = self.max_length - len(input_ids)
        
        if padding_length > 0:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            if self.padding_side == "left":
                input_ids = [pad_token_id] * padding_length + input_ids
                labels = [-100] * padding_length + labels
                attention_mask = [0] * padding_length + [1] * len(example['input_ids'])
            else:
                input_ids = input_ids + [pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                attention_mask = [1] * len(example['input_ids']) + [0] * padding_length
        elif padding_length < 0:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def load_conversations(json_path: str) -> List[Dict]:
    """Load conversations from JSON file.
    
    Args:
        json_path: Path to conversations.json file
        
    Returns:
        List of conversation dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    return conversations


def create_datasets(
    conversations_json_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    outgoing_speaker_name: str = "Ryan Amiri",
    padding_side: str = "left"
) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset]:
    """Create train/val/test datasets from conversations JSON.
    
    Args:
        conversations_json_path: Path to conversations.json
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for test
        random_seed: Random seed for splitting
        outgoing_speaker_name: Name for outgoing messages
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    conversations = load_conversations(conversations_json_path)
    
    train_convos, val_convos, test_convos = split_conversations(
        conversations,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    
    print(f"Loaded {len(conversations):,} conversations")
    print(f"Train: {len(train_convos):,}, Val: {len(val_convos):,}, Test: {len(test_convos):,}")
    
    train_examples = build_training_examples_from_conversations(
        train_convos, tokenizer, max_length, outgoing_speaker_name
    )
    val_examples = build_training_examples_from_conversations(
        val_convos, tokenizer, max_length, outgoing_speaker_name
    )
    test_examples = build_training_examples_from_conversations(
        test_convos, tokenizer, max_length, outgoing_speaker_name
    )
    
    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")
    print(f"Test examples: {len(test_examples):,}")
    
    train_dataset = ConversationDataset(train_examples, tokenizer, max_length, padding_side)
    val_dataset = ConversationDataset(val_examples, tokenizer, max_length, padding_side)
    test_dataset = ConversationDataset(test_examples, tokenizer, max_length, padding_side)
    
    return train_dataset, val_dataset, test_dataset

