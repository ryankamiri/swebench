#!/bin/bash
# Setup script for SWE-bench with Qwen2.5-Coder-7B and custom dataset

echo "Setting up SWE-bench environment with Qwen2.5-Coder-7B support..."

# Create conda environment
echo "Creating conda environment 'swebench-qwen'..."
conda create -n swebench-qwen python=3.9 -y

# Activate environment
echo "Activating environment..."
conda activate swebench-qwen

# Install basic dependencies
echo "Installing basic dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.30.0
pip install datasets>=2.0.0
pip install accelerate
pip install bitsandbytes
pip install wandb
pip install tqdm
pip install requests
pip install psutil

# Install SWE-bench in development mode
echo "Installing SWE-bench..."
pip install -e .

# Install additional dependencies for the custom setup
echo "Installing additional dependencies..."
pip install git+https://github.com/huggingface/transformers.git
pip install flash-attn --no-build-isolation

# Install Apptainer dependencies (if using Apptainer instead of Docker)
echo "Installing Apptainer dependencies..."
pip install -r requirements-apptainer.txt

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate swebench-qwen"
echo ""
echo "To test the setup, run:"
echo "python test_setup.py"
