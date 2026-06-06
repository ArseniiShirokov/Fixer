#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .inference_pretrained_model import (
    get_image_paths,
    get_resolution_size,
    load_and_compile_model,
    model_inference,
    postprocess_output,
    preprocess_image,
)


DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class FixerInferencer:

    def __init__(
        self,
        model_path: str,
        timestep: int = 250,
        resolution: int = 1024,
        dtype: str = "bf16",
        device: str = "cuda",
        compile_model: bool = True,
        batch_size: int = 1,
        warmup_iters: int = 10,
    ) -> None:
        self.model_path = model_path
        self.timestep = timestep
        self.resolution = resolution
        self.dtype = DTYPE_BY_NAME[dtype]
        self.device = torch.device(device)
        self.compile_model = compile_model
        self.batch_size = batch_size
        self.warmup_iters = warmup_iters

        self.h, self.w = 1024, 576
        self.input_size = get_resolution_size(resolution)
        self.model = None
        self._warmup_tensor = None

    def prepare_model(self) -> None:
        self.model = load_and_compile_model(
            model_path=self.model_path,
            timestep=self.timestep,
            vae_skip_connection=False,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
            compile=self.compile_model,
        )

        self._warmup_tensor = torch.randn(
            self.batch_size,
            3,
            self.h,
            self.w,
            device=self.device,
            dtype=self.dtype,
        )

        with torch.no_grad():
            for _ in tqdm(range(self.warmup_iters), desc="Warmup", leave=False):
                model_inference(
                    self.model,
                    self.batch_size,
                    self.h,
                    self.w,
                    self.dtype,
                    self.device,
                    x=self._warmup_tensor,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _preprocess(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        original_size = image.size
        x = preprocess_image(image.resize(self.input_size, Image.BILINEAR), self.device, self.dtype)
        return x, original_size

    def _run_inference(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not prepared. Call prepare_model() first.")
        with torch.no_grad():
            return model_inference(
                self.model,
                self.batch_size,
                self.h,
                self.w,
                self.dtype,
                self.device,
                x=x,
            )

    def infer(self, image: Image.Image) -> Image.Image:
        x, original_size = self._preprocess(image)
        output = self._run_inference(x)
        return postprocess_output(output, original_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Fixer inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--resolution", type=int, default=1024, choices=[256, 512, 704, 960, 1024, 1360])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = FixerInferencer(
        model_path=args.model,
        timestep=args.timestep,
        resolution=args.resolution,
        dtype=args.dtype,
        device=args.device,
        compile_model=not args.no_compile,
        batch_size=1,
        warmup_iters=args.warmup_iters,
    )
    inferencer.prepare_model()

    image_paths = get_image_paths(args.input, max_frames=1, skip_frames=1)

    image = Image.open(image_paths[0]).convert("RGB")
    result = inferencer.infer(image)
    out_path = output_dir / Path(image_paths[0]).name
    result.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
