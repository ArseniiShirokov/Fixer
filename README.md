# Fixer

## Full setup and minimal inference

### 1) Create and activate conda environment

```bash
conda create -n fixer_env python=3.12.3 -y
conda activate fixer_env
```

### 2) Install CUDA

```bash
conda install -y -c "nvidia/label/cuda-12.9.0" \
  cuda-cudart cuda-cudart-dev cuda-nvrtc cuda-libraries cuda-libraries-dev \
  cuda-nvcc libcublas libcublas-dev cudnn=9 cuda-nvtx cuda-nvtx-dev cuda-nvml-dev
```

### 3) Upgrade pip tooling

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4) Install Torch

```bash
python -m pip install "torch==2.7.0"
```

### 5) Install Transformer Engine 

```bash
python -m pip install --no-build-isolation --extra-index-url https://pypi.nvidia.com \
  "transformer-engine==2.2.0" \
  "transformer-engine-cu12==2.2.0" \
  "transformer-engine-torch==2.2.0"
```

### 6) Install dependencies 

```bash
python -m pip install --no-deps -r fixer_requirements.txt
```

### 7) Install Fixer package from repository

```bash
python -m pip install --no-deps .
```

### 8) Download model weights

```bash
pip install "huggingface_hub[cli]"
hf auth login
hf download nvidia/Fixer --local-dir models
```

### 9) Minimal run on one image from `example`

```bash
python src/infer.py \
  --model models/pretrained/pretrained_fixer.pkl \
  --input example \
  --output output \
  --timestep 250 \
  --resolution 1024 \
  --dtype bf16
```
