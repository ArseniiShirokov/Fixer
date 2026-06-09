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
        use_cuda_graph: bool = True,
    ) -> None:
        self.model_path = model_path
        self.timestep = timestep
        self.resolution = resolution
        self.dtype = DTYPE_BY_NAME[dtype]
        self.device = torch.device(device)
        self.compile_model = compile_model
        self.batch_size = batch_size
        self.warmup_iters = warmup_iters
        self.use_cuda_graph = use_cuda_graph

        self.h, self.w = 1024, 576
        self.input_size = get_resolution_size(resolution)
        self.model = None
        self._warmup_tensor = None

        # CUDA-graph state. The diffusion input shape is constant, so the whole
        # forward is captured once and replayed, eliminating the per-kernel
        # launch gaps that leave the GPU idle at small batch sizes.
        self._graph = None
        self._static_in = None
        self._static_out = None

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

        # Match the real preprocessed input shape: an image resized to
        # input_size=(W, H) becomes a tensor [B, 3, H, W]. (h/w are vestigial
        # and were transposed relative to the actual input.)
        in_h, in_w = self.input_size[1], self.input_size[0]
        self._warmup_tensor = torch.randn(
            self.batch_size,
            3,
            in_h,
            in_w,
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

        if self.use_cuda_graph and self.device.type == "cuda":
            try:
                self._capture_cuda_graph()
            except Exception as e:  # capture can fail on non-capturable ops
                print(f"[cuda-graph] capture failed ({type(e).__name__}: {e}); "
                      f"falling back to non-graph path")
                self._graph = None

    def _autocast(self):
        # cache_enabled=False: autocast's cast-weight cache is unsafe across
        # CUDA-graph capture/replay; weights are already in self.dtype anyway.
        return torch.autocast(device_type="cuda", dtype=self.dtype, cache_enabled=False)

    def _capture_cuda_graph(self) -> None:
        """Capture the full forward (vae enc -> DiT denoise -> vae dec) into one graph."""
        in_h, in_w = self.input_size[1], self.input_size[0]
        self._static_in = torch.randn(
            self.batch_size, 3, in_h, in_w, device=self.device, dtype=self.dtype
        )
        # Required warmup on a side stream before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with torch.no_grad(), self._autocast():
                for _ in range(3):
                    self._static_out = self.model(self._static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize(self.device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), self._autocast():
            with torch.cuda.graph(self._graph):
                self._static_out = self.model(self._static_in)
        torch.cuda.synchronize(self.device)
        _, _, sh, sw = self._static_in.shape
        print(f"[cuda-graph] captured forward for batch_size={self.batch_size} "
              f"@ {sh}x{sw}")

    def _preprocess(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        original_size = image.size
        x = preprocess_image(image.resize(self.input_size, Image.BILINEAR), self.device, self.dtype)
        return x, original_size

    def _run_inference(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not prepared. Call prepare_model() first.")
        if self._graph is not None:
            # Copy the new frame into the captured input buffer and replay.
            # _static_out is valid until the next replay; postprocess reads it
            # (and moves to CPU) immediately, so no clone is needed here.
            self._static_in.copy_(x)
            self._graph.replay()
            return self._static_out
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
    parser.add_argument("--no-cuda-graph", action="store_true")
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
        use_cuda_graph=not args.no_cuda_graph,
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
