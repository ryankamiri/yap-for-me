# YapForMe

A project to fine-tune a language model to text like you, based on your iMessage history.

## Overview

YapForMe processes your exported iMessage conversations and fine-tunes a language model to generate responses in your personal texting style. The model learns from your actual message history to capture your tone, phrasing, and communication patterns.

## Project Status

This project is currently in development. The core components are implemented, but training and evaluation are ongoing.

## Features

- Preprocesses iMessage CSV exports from iMazing 📱
- Filters and cleans conversation data 🧹
- Tokenizes conversations for efficient training ⚡
- Fine-tunes language models using PyTorch 🚀
- Supports training on HPC clusters with SLURM 🖥️
- Tracks training metrics with Weights & Biases 📊

## Setup

### 1. Environment Setup

Create and activate the conda environment:

```bash
./scripts/setup_env.sh
conda activate yap
```

Or manually:

```bash
conda create -n yap python=3.11 -y
conda activate yap
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### 2. Data Preparation

1. Export your iMessage history from iMazing as a CSV file
2. Place it in `training/data/` (e.g., `training/data/messages.csv`)
3. Update the path in the preprocessing notebook
4. Run the preprocessing notebook: `training/notebooks/00 dataset generation.ipynb`
   - This filters messages, groups conversations, and creates `training/data/conversations.json`

### 3. Configuration

Edit `training/configs/config.yaml` to set:
- Model name and parameters
- Dataset paths
- Training hyperparameters (epochs, batch size, learning rate, etc.)

## Usage

### Preprocessing (Tokenization)

Tokenize all conversations and save them for faster training:

```bash
cd training
python3 preprocess_dataset.py --config configs/config.yaml
```

Or submit to SLURM:

```bash
sbatch preprocess.sbatch
```

This creates `data/tokenized_examples.pt` with all pre-tokenized training examples.

### Training

Train the model:

```bash
cd training
python3 train.py --config configs/config.yaml
```

Or submit to SLURM:

```bash
sbatch yap.sbatch
```

Training outputs (checkpoints, logs) are saved to `out/<job_id>/`.

## Configuration

Key configuration options in `training/configs/config.yaml`:

- `model.name`: Hugging Face model identifier
- `model.max_length`: Maximum sequence length (4096)
- `dataset.tokenized_examples_path`: Path to pre-tokenized examples
- `dataset.outgoing_speaker_name`: Name for your outgoing messages
- `training.epochs`: Number of training epochs
- `training.batch_size`: Batch size
- `training.learning_rate`: Learning rate

## Training Details

- **Objective**: Next-token prediction on your outgoing messages 🎯
- **Context**: Uses previous conversation history as context 💬
- **Format**: Only predicts message text (not timestamps, speakers, or reply metadata) ✍️
- **Loss**: Cross-entropy loss on target tokens only 📉

## Data Processing

The preprocessing pipeline:

1. Loads iMessage CSV export 📥
2. Filters out:
   - Empty messages
   - Verification codes (numeric only)
   - Outgoing messages with excluded words
   - Outgoing messages with attachments
   - Conversations with no outgoing messages
3. Groups messages by chat session 👥
4. Sorts messages chronologically ⏰
5. Creates training examples where:
   - Context: Previous messages in the conversation
   - Target: Your next outgoing message (text only)

## Dependencies

See `requirements.txt` for Python dependencies. Key packages:
- PyTorch (with CUDA support)
- Transformers (Hugging Face)
- Wandb (experiment tracking)
- Pandas, PyYAML

## Notes

- The model only learns to generate the text content of messages, not metadata 📝
- Reply information is included in context but not predicted 🔗
- Training examples are split randomly (not by conversation) since they're independent once tokenized 🎲

## Backend

The backend is a FastAPI server that integrates with BlueBubbles (local iMessage server) and a GPU server hosting the language model.

### Architecture

- **BlueBubbles Server**: Local macOS server for iMessage integration
- **FastAPI Backend**: Handles webhooks, manages conversation context, and orchestrates actions
- **GPU Server**: Hosts the LLM for inference (supports vLLM, Ollama, TGI)

### Flow

1. BlueBubbles webhook receives a new incoming message
2. Backend fetches conversation history (if not cached) and populates context window
3. Model inference is performed with conversation context
4. **TEMPORARY**: Model output is sent as plain text to the sender
5. **TODO**: Replace with proper tool calling extraction to parse and execute structured actions

### Configuration

Backend configuration is managed through:
- **Environment Variables** (`.env`): BlueBubbles URL/password, GPU server URL/API key, backend host/port
- **Inference Config** (`backend/inference_config.yaml`): Model parameters (temperature, max_tokens, etc.)

### Running the Backend

```bash
cd backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Important Notes

⚠️ **Temporary Solution**: Currently, the model output is sent directly as text to the sender. This is a temporary implementation that needs to be replaced with efficient tool calling extraction. The model should output structured actions (send, reply, react, etc.) that are then parsed and executed appropriately.

## Future Work

- Model evaluation and generation testing 🧪
- Integration with phone/texting interface 📲
- **Tool calling extraction and execution** (replaces temporary text sending) 🛠️
- Additional filtering and data quality improvements ✨

