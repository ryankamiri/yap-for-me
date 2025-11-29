import json
from pathlib import Path
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def format_message_prefix(timestamp: str, speaker: str, replying_to: str = None, guid: str = None) -> str:
    """Format just the prefix part of a message (everything except the text).
    Returns: [guid:xxx][timestamp] speaker: [replying_to ] (without the actual text)
    """
    guid_part = f"[guid:{guid}]" if guid else ""
    if replying_to:
        return f"{guid_part}[{timestamp}] {speaker}: {replying_to} "
    return f"{guid_part}[{timestamp}] {speaker}: "


def format_message(timestamp: str, speaker: str, text: str, replying_to: str = None, guid: str = None) -> str:
    """Format a message into the standard format: [guid:xxx][timestamp] Speaker: text
    If replying_to is provided, it's included before the text.
    """
    return format_message_prefix(timestamp, speaker, replying_to, guid) + text


def has_incoming_context(messages: List[Dict], current_idx: int) -> bool:
    """Check if there's at least one Incoming message before the current index."""
    for i in range(current_idx):
        if messages[i]['type'] == 'Incoming':
            return True
    return False


def group_consecutive_outgoing(messages: List[Dict]) -> List[List[int]]:
    """Group consecutive Outgoing message indices.
    
    Args:
        messages: List of message dicts
        
    Returns:
        List of groups, each group is a list of message indices
    """
    groups = []
    current_group = []
    
    for idx, msg in enumerate(messages):
        if msg['type'] == 'Outgoing':
            current_group.append(idx)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
    
    if current_group:
        groups.append(current_group)
    
    return groups


def build_training_examples_from_conversations(
    conversations: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length
) -> List[Dict]:
    """Build training examples from conversations.
    
    Optimized version: Pre-tokenizes messages once per conversation, then builds
    examples by slicing pre-tokenized token arrays. Uses batch tokenization and
    numpy for efficient operations.
    
    Groups consecutive Outgoing messages and creates training examples:
    - Context: All messages before the first Outgoing in the group (uses text field)
    - Target: Combined tool calls from all Outgoing messages in the group
    - Context tokens get -100 labels, target tokens get normal labels
    
    Args:
        conversations: List of conversation dicts from JSON
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of training examples with 'input_ids' and 'labels'
    """
    examples = []
    
    for _, conversation in enumerate(conversations):
        messages = conversation['messages']
        
        # Early filter: Skip conversations with no outgoing messages
        if not any(msg['type'] == 'Outgoing' for msg in messages):
            continue
        
        # Phase 1: Pre-tokenization (once per conversation)
        
        # Compute outgoing groups first
        outgoing_groups = group_consecutive_outgoing(messages)
        if not outgoing_groups:
            continue
        
        # Format all messages once (with trailing newlines)
        formatted_messages = []
        for msg in messages:
            formatted = format_message(
                msg['timestamp'],
                msg['speaker'],
                msg['text'],
                msg['replying_to'],
                msg['guid']
            )
            formatted_with_newline = formatted + '\n'
            formatted_messages.append(formatted_with_newline)
        
        # Batch tokenize all formatted messages
        tokenized_results = tokenizer(
            formatted_messages,
            add_special_tokens=False,
            return_attention_mask=False
        )['input_ids']
        pre_tokenized_messages = [np.array(tokens, dtype=np.int64) for tokens in tokenized_results]
        
        # Collect all tool call texts for all outgoing groups
        tool_call_texts = []
        valid_groups = []
        for group in outgoing_groups:
            if not group:
                continue
            
            first_outgoing_idx = group[0]
            if not has_incoming_context(messages, first_outgoing_idx):
                continue
            
            tool_calls = []
            for msg_idx in group:
                msg = messages[msg_idx]
                tool_call = msg.get('tool_call')
                if tool_call:
                    tool_calls.append(tool_call)
            
            if tool_calls:
                tool_call_texts.append('\n'.join(tool_calls))
                valid_groups.append(group)
        
        if not tool_call_texts:
            continue
        
        # Batch tokenize all tool call texts
        tool_call_tokenized = tokenizer(
            tool_call_texts,
            add_special_tokens=False,
            return_attention_mask=False
        )['input_ids']
        pre_tokenized_tool_calls = [np.array(tokens, dtype=np.int64) for tokens in tool_call_tokenized]
        
        # Phase 2: Build examples using pre-tokenized data
        
        # Pre-concatenate ALL messages once
        full_context_tokens = np.concatenate(pre_tokenized_messages)
        
        # Pre-compute cumulative token counts for each message boundary
        # This tells us where each message ends in the concatenated array
        cumulative_lengths = [0]
        for msg_tokens in pre_tokenized_messages:
            cumulative_lengths.append(cumulative_lengths[-1] + len(msg_tokens))
        
        for group, tool_call_tokens in zip(valid_groups, pre_tokenized_tool_calls):
            first_outgoing_idx = group[0]
            
            if first_outgoing_idx == 0:
                continue
            
            # Slice directly from pre-concatenated array 
            # cumulative_lengths[first_outgoing_idx] gives us the end position
            context_tokens = full_context_tokens[:cumulative_lengths[first_outgoing_idx]]
            
            # Truncate context if needed (add-then-truncate approach)
            total_length = len(context_tokens) + len(tool_call_tokens)
            if total_length > max_length:
                available_context = max_length - len(tool_call_tokens)
                if available_context <= 0:
                    continue
                context_tokens = context_tokens[-available_context:]
            
            # Handle tool call truncation if still needed
            final_total = len(context_tokens) + len(tool_call_tokens)
            if final_total > max_length:
                available_for_tool_calls = max_length - len(context_tokens)
                if available_for_tool_calls <= 0:
                    continue
                tool_call_tokens = tool_call_tokens[:available_for_tool_calls]
            
            # Combine context and tool calls
            input_ids = np.concatenate([context_tokens, tool_call_tokens])
            
            # Create labels: -100 for context, normal labels for tool calls
            labels = np.concatenate([
                np.full(len(context_tokens), -100, dtype=np.int64),
                tool_call_tokens
            ])
            
            # Convert to Python lists for storage
            examples.append({
                'input_ids': input_ids.tolist(),
                'labels': labels.tolist(),
                'context_length': len(context_tokens),
                'target_length': len(tool_call_tokens)
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


def load_tokenized_examples(torch_path: str) -> List[Dict]:
    """Load pre-tokenized examples from torch file.
    
    Args:
        torch_path: Path to .pt file with tokenized examples (saved with torch.save)
        
    Returns:
        List of training examples with 'input_ids' and 'labels'
        
    Raises:
        FileNotFoundError: If the tokenized examples file doesn't exist
    """
    path = Path(torch_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenized examples file not found: {torch_path}\n"
            f"Please run preprocess_dataset.py first to create the tokenized examples."
        )
    
    examples = torch.load(torch_path, map_location='cpu')
    return examples


def create_datasets(
    tokenized_examples_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    padding_side: str = "left"
) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset]:
    """Create train/val/test datasets from pre-tokenized examples.
    
    Args:
        tokenized_examples_path: Path to .pt file with pre-tokenized examples
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for test
        random_seed: Random seed for splitting
        padding_side: "left" or "right" - where to pad sequences
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        FileNotFoundError: If the tokenized examples file doesn't exist
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    print(f"Loading pre-tokenized examples from: {tokenized_examples_path}")
    all_examples = load_tokenized_examples(tokenized_examples_path)
    print(f"Loaded {len(all_examples)} examples")
    
    # Shuffle examples and split
    random.seed(random_seed)
    shuffled = all_examples.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_examples = shuffled[:train_end]
    val_examples = shuffled[train_end:val_end]
    test_examples = shuffled[val_end:]
    
    print(f"Train: {len(train_examples):,}, Val: {len(val_examples):,}, Test: {len(test_examples):,}")
    
    train_dataset = ConversationDataset(train_examples, tokenizer, max_length, padding_side)
    val_dataset = ConversationDataset(val_examples, tokenizer, max_length, padding_side)
    test_dataset = ConversationDataset(test_examples, tokenizer, max_length, padding_side)
    
    return train_dataset, val_dataset, test_dataset

