#!/bin/bash
# Quick setup script for SWE-bench with Qwen2.5-Coder-7B

set -e  # Exit on any error

echo "🚀 Setting up SWE-bench with Qwen2.5-Coder-7B"
echo "=============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install conda first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create conda environment
echo "📦 Creating conda environment 'swebench-qwen'..."
conda create -n swebench-qwen python=3.9 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swebench-qwen

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
# For CUDA 12.x, use the cu121 index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📚 Installing core dependencies..."
# Fix importlib-metadata for Python 3.9 compatibility
pip install --upgrade importlib-metadata
pip install transformers>=4.30.0
pip install datasets>=2.0.0
pip install accelerate
pip install bitsandbytes
pip install wandb
pip install tqdm
pip install requests
pip install psutil

# Install SWE-bench
echo "🔧 Installing SWE-bench..."
pip install -e .

# Install additional dependencies
echo "⚡ Installing additional dependencies..."
pip install git+https://github.com/huggingface/transformers.git

# Install flash-attn compatible with CUDA 12.x
echo "🔥 Installing flash-attn (CUDA 12.x compatible)..."
pip install flash-attn --no-build-isolation || {
    echo "⚠️  Standard flash-attn installation failed. Trying alternative..."
    pip install ninja packaging
    pip install flash-attn==2.7.4 --no-build-isolation || echo "⚠️  flash-attn installation failed (this is optional)"
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate swebench-qwen"
echo ""
echo "To test the setup, run:"
echo "python test_setup.py"
echo ""
echo "To explore the dataset, run:"
echo "python explore_dataset.py"
echo ""
echo "To run the complete pipeline, run:"
echo "python run_qwen_evaluation.py --max_instances 3 --use_quantization"
echo ""
echo "Don't forget to login to wandb:"
echo "wandb login"
