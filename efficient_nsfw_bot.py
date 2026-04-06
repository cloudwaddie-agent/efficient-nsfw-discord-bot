#!/usr/bin/env python3
"""Super efficient Discord NSFW image detection bot."""

import os
import gc
import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
import requests
from io import BytesIO
import discord
from discord.ext import commands

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    discord_token: str = os.getenv("DISCORD_TOKEN", "")
    model_name: str = "Marqo/nsfw-image-detection-384"
    device: str = "cpu"
    nsfw_threshold: float = 0.7
    max_image_size: int = 2048
    clear_cache_interval: int = 10
    max_concurrent_checks: int = 3
    request_timeout: int = 10
    max_retries: int = 2


config = Config()


class MemoryManager:
    def __init__(self):
        self.count = 0

    def should_clean(self) -> bool:
        self.count += 1
        return self.count % config.clear_cache_interval == 0

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.count = 0


mem = MemoryManager()


class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self._onnx = False

    async def load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForImageClassification, AutoImageProcessor

        logger.info(f"Loading {config.model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)

        try:
            from optimum.onnxruntime import ORTModelForImageClassification
            logger.info("Using ONNX Runtime (much faster on CPU)")
            self.model = ORTModelForImageClassification.from_pretrained(
                config.model_name,
                export=True
            )
            self._onnx = True
        except Exception as e:
            logger.info(f"ONNX not available, using PyTorch: {e}")
            self.model = AutoModelForImageClassification.from_pretrained(config.model_name)
            self.model.to(config.device)
            self.model.eval()

            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("torch.compile enabled")
                except Exception:
                    pass
            self._onnx = False

        logger.info("Model loaded")

    async def predict(self, img: Image.Image) -> float:
        if self.model is None:
            await self.load()

        if self._onnx:
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**inputs)
                probs = torch.softmax(out.logits, dim=-1)
        else:
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                probs = torch.softmax(out.logits, dim=-1)

        return probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()

    def unload(self):
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            mem.cleanup()


m = ModelManager()


class ImageHandler:
    @staticmethod
    async def get(url: str) -> Optional[Image.Image]:
        for _ in range(config.max_retries):
            try:
                r = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: requests.get(url, timeout=config.request_timeout)
                    ),
                    timeout=config.request_timeout + 5
                )
                r.raise_for_status()
                img = Image.open(BytesIO(r.content))

                if img.mode != "RGB":
                    img = img.convert("RGB")

                if max(img.size) > config.max_image_size:
                    ratio = config.max_image_size / max(img.size)
                    img = img.resize(tuple(int(d * ratio) for d in img.size), Image.LANCZOS)

                return img
            except Exception as e:
                logger.warning(f"Download failed: {e}")
        return None

    @staticmethod
    def valid(fname: str) -> bool:
        return fname.lower().split('.')[-1] in ('jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp')


class Bot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix="!",
            intents=discord.Intents(messages=True, message_content=True, guilds=True),
            max_messages=100,
            asyncio_debug=False
        )
        self.sem = asyncio.Semaphore(config.max_concurrent_checks)

    async def setup_hook(self):
        await m.load()
        logger.info("Bot ready")

    async def close(self):
        m.unload()
        await super().close()

    async def on_message(self, msg: discord.Message):
        if msg.author.bot or msg.author.system or not msg.attachments:
            return

        for att in msg.attachments:
            if not ImageHandler.valid(att.filename):
                continue

            asyncio.create_task(self._check(msg, att))
            if mem.should_clean():
                mem.cleanup()

    async def _check(self, msg: discord.Message, att: discord.Attachment):
        async with self.sem:
            try:
                img = await ImageHandler.get(att.url)
                if img is None:
                    return

                score = await m.predict(img)
                logger.info(f"{att.filename}: {score:.3f}")

                if score >= config.nsfw_threshold:
                    await msg.delete()
                    logger.info(f"Deleted: {att.filename} ({score:.2%})")

                    try:
                        await msg.channel.send(
                            f"Deleted NSFW from {msg.author.mention}",
                            delete_after=5.0
                        )
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error: {e}")


async def main():
    if not config.discord_token:
        logger.error("Set DISCORD_TOKEN env var!")
        return

    bot = Bot()
    try:
        await bot.start(config.discord_token)
    except KeyboardInterrupt:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())