"""OpenCLIP adapter. Float32 inference, normalized embeddings, explicit device."""
from __future__ import annotations

import importlib.metadata
import json
import hashlib
from pathlib import Path
from threading import RLock

import numpy as np

from .data import load_image

MODELS = {
    "b32-openai": ("ViT-B-32", "openai"),
    "b32-laion": ("ViT-B-32", "laion2b_s34b_b79k"),
    "b16-openai": ("ViT-B-16", "openai"),
}
PROMPTS = ("{}", "an image of {}", "a photo of {}")


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norm < 1e-8).any():
        raise ValueError("Embeddings must be finite and nonzero")
    return x / norm


class OpenCLIPEncoder:
    def __init__(self, model: str = "b32-openai", device: str = "auto", cache: str = ".cache/models"):
        import torch
        import open_clip

        if model not in MODELS:
            raise ValueError(f"Unknown model {model}; choose {list(MODELS)}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device, self.model_key = device, model
        self.lock = RLock()
        architecture, pretrained = MODELS[model]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            architecture, pretrained=pretrained, device="cpu", precision="fp32", cache_dir=cache,
            force_quick_gelu=(pretrained == "openai"),
        )
        self.model.eval()
        weights = hashlib.sha256()
        for name, parameter in sorted(self.model.state_dict().items()):
            weights.update(name.encode())
            weights.update(parameter.detach().contiguous().numpy().tobytes())
        self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(architecture)
        self.info = {
            "backend": "open_clip", "model": model, "architecture": architecture,
            "pretrained": pretrained, "open_clip": importlib.metadata.version("open_clip_torch"),
            "precision": "fp32", "preprocess": repr(self.preprocess),
            "weights_sha256": weights.hexdigest(),
            "quick_gelu": pretrained == "openai",
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.info, sort_keys=True).encode()).hexdigest()

    def _numpy(self, tensor) -> np.ndarray:
        # CPU transfer synchronizes CUDA/MPS before latency is recorded.
        return normalize(tensor.detach().float().cpu().numpy())

    def texts(self, texts: list[str], ensemble: bool = False, batch_size: int = 32) -> np.ndarray:
        import torch
        if not texts or batch_size < 1:
            raise ValueError("Nonempty texts and a positive batch size are required")
        expanded = [p.format(t) for t in texts for p in PROMPTS] if ensemble else texts
        chunks = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(expanded), batch_size):
                tokens = self.tokenizer(expanded[start:start + batch_size]).to(self.device)
                chunks.append(self._numpy(self.model.encode_text(tokens)))
        result = np.concatenate(chunks)
        if ensemble:
            result = normalize(result.reshape(len(texts), len(PROMPTS), -1).mean(axis=1))
        return result

    def images(self, images: list, batch_size: int = 8) -> np.ndarray:
        import torch
        if not images or batch_size < 1:
            raise ValueError("Nonempty images and a positive batch size are required")
        chunks = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = [load_image(x) if isinstance(x, (str, Path)) else x for x in images[start:start + batch_size]]
                tensor = torch.stack([self.preprocess(x) for x in batch]).to(self.device)
                chunks.append(self._numpy(self.model.encode_image(tensor)))
        return np.concatenate(chunks)
