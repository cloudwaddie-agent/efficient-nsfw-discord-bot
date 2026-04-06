# Efficient NSFW Discord Bot

CPU-optimized Discord bot that detects and deletes NSFW images using Marqo/nsfw-image-detection-384 model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install ONNX for faster CPU inference
pip install optimum[onnxruntime]

# Run
export DISCORD_TOKEN="your_bot_token_here"
python efficient_nsfw_bot.py
```

## Requirements

- Python 3.8+
- PyTorch (for CPU inference)
- Transformers (HuggingFace)
- discord.py

## Features

- **CPU Optimized**: Runs efficiently on any CPU
- **Memory Efficient**: Periodic garbage collection, model cleanup on shutdown
- **ONNX Support**: Auto-detects and uses ONNX Runtime if available (much faster)
- **torch.compile**: Uses PyTorch 2.0+ compilation when available
- **Async Processing**: Concurrent image downloads with semaphore limiting
- **Auto-resize**: Large images automatically resized to save memory

## Configuration

Edit `Config` class in the code to adjust:
- `nsfw_threshold`: Detection sensitivity (default 0.7)
- `max_image_size`: Max image dimension (default 2048)
- `clear_cache_interval`: GC frequency (default 10)
- `max_concurrent_checks`: Parallel download limit (default 3)