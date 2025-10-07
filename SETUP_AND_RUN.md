# SWE-bench with Qwen2.5-Coder-7B Setup and Run Guide

This guide provides complete instructions for setting up and running SWE-bench with the Qwen2.5-Coder-7B model on the custom dataset `ryankamiri/SWE-bench_oracle_lite` with wandb logging.

## Prerequisites

- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- Conda or Miniconda installed
- Docker installed and running
- Wandb account (for logging)

## Step 1: Environment Setup

### Create and Activate Conda Environment

```bash
# Create conda environment
conda create -n swebench-qwen python=3.9 -y

# Activate environment
conda activate swebench-qwen
```

### Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers>=4.30.0
pip install datasets>=2.0.0
pip install accelerate
pip install bitsandbytes
pip install wandb
pip install tqdm
pip install requests
pip install psutil

# Install SWE-bench in development mode
pip install -e .

# Install additional dependencies
pip install git+https://github.com/huggingface/transformers.git
pip install flash-attn --no-build-isolation
```

### Setup Wandb

```bash
# Login to wandb (you'll need your API key)
wandb login
```

## Step 2: Test the Setup

```bash
# Test that everything is working
python test_setup.py
```

This will verify:
- All packages are installed correctly
- GPU is available
- Model can be loaded
- Dataset can be loaded
- Wandb is configured

## Step 3: Explore the Dataset

```bash
# Explore the custom dataset structure
python explore_dataset.py
```

This will show you the dataset structure and save a sample to `dataset_sample.json`.

## Step 4: Run the Complete Pipeline

### Option 1: Full Pipeline (Inference + Evaluation)

```bash
# Run complete pipeline with default settings
python run_qwen_evaluation.py \
    --dataset_name "ryankamiri/SWE-bench_oracle_lite" \
    --model_name "Qwen/Qwen2.5-Coder-7B" \
    --wandb_project "swebench-qwen" \
    --wandb_run_name "qwen_oracle_lite_full"
```

### Option 2: With Custom Settings

```bash
# Run with custom settings
python run_qwen_evaluation.py \
    --dataset_name "ryankamiri/SWE-bench_oracle_lite" \
    --model_name "Qwen/Qwen2.5-Coder-7B" \
    --max_instances 10 \
    --use_quantization \
    --max_workers 2 \
    --wandb_project "swebench-qwen" \
    --wandb_run_name "qwen_oracle_lite_test"
```

### Option 3: Test with Specific Instances

```bash
# Run on specific instances only
python run_qwen_evaluation.py \
    --dataset_name "ryankamiri/SWE-bench_oracle_lite" \
    --model_name "Qwen/Qwen2.5-Coder-7B" \
    --instance_ids "instance1" "instance2" "instance3" \
    --wandb_project "swebench-qwen" \
    --wandb_run_name "qwen_specific_instances"
```

## Step 5: Run Individual Components

### Only Inference (Generate Predictions)

```bash
python run_qwen_evaluation.py \
    --dataset_name "ryankamiri/SWE-bench_oracle_lite" \
    --model_name "Qwen/Qwen2.5-Coder-7B" \
    --skip_evaluation \
    --wandb_project "swebench-qwen"
```

### Only Evaluation (Use Existing Predictions)

```bash
python run_qwen_evaluation.py \
    --predictions_path "predictions_qwen.jsonl" \
    --dataset_name "ryankamiri/SWE-bench_oracle_lite" \
    --skip_inference \
    --wandb_project "swebench-qwen"
```

## Step 6: Monitor Results

### Wandb Dashboard

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project (default: `swebench-qwen`)
3. View real-time metrics and results

### Local Results

- **Predictions**: `predictions_qwen.jsonl`
- **Evaluation Report**: `evaluation_results/{run_id}_report.json`
- **Logs**: `logs/` directory

## Command Line Options

### Dataset and Model Options

- `--dataset_name`: Dataset to use (default: `ryankamiri/SWE-bench_oracle_lite`)
- `--model_name`: Model to use (default: `Qwen/Qwen2.5-Coder-7B`)
- `--max_instances`: Limit number of instances to process
- `--instance_ids`: Process specific instances only
- `--use_quantization`: Use 4-bit quantization to reduce memory usage

### Evaluation Options

- `--max_workers`: Number of parallel workers (default: 4)
- `--timeout`: Timeout for tests in seconds
- `--rm_image`: Remove Docker images after evaluation
- `--force_rebuild`: Force rebuild all Docker images
- `--cache_level`: Image caching level (`none`, `base`, `env`, `instance`)
- `--clean`: Clean cached images

### Wandb Options

- `--wandb_project`: Wandb project name (default: `swebench-qwen`)
- `--wandb_run_name`: Wandb run name (auto-generated if not provided)

### Pipeline Control

- `--skip_inference`: Skip inference phase
- `--skip_evaluation`: Skip evaluation phase
- `--run_id`: Custom run ID

## Memory Requirements

### Without Quantization
- **GPU Memory**: ~14GB VRAM
- **System RAM**: ~16GB
- **Disk Space**: ~120GB

### With Quantization (`--use_quantization`)
- **GPU Memory**: ~8GB VRAM
- **System RAM**: ~16GB
- **Disk Space**: ~120GB

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use quantization
   python run_qwen_evaluation.py --use_quantization
   
   # Or reduce batch size
   python run_qwen_evaluation.py --max_instances 5
   ```

2. **Docker Issues**
   ```bash
   # Check Docker is running
   docker --version
   
   # Restart Docker if needed
   sudo systemctl restart docker
   ```

3. **Wandb Login Issues**
   ```bash
   # Re-login to wandb
   wandb login
   ```

4. **Dataset Loading Issues**
   ```bash
   # Test dataset loading
   python explore_dataset.py
   ```

### Performance Optimization

1. **For Limited GPU Memory**:
   ```bash
   python run_qwen_evaluation.py --use_quantization --max_instances 10
   ```

2. **For Faster Evaluation**:
   ```bash
   python run_qwen_evaluation.py --max_workers 8 --cache_level instance
   ```

3. **For Testing**:
   ```bash
   python run_qwen_evaluation.py --max_instances 3 --max_workers 1
   ```

## Example Commands

### Quick Test (3 instances, quantized)
```bash
python run_qwen_evaluation.py \
    --max_instances 3 \
    --use_quantization \
    --max_workers 1 \
    --wandb_project "swebench-qwen-test"
```

### Full Run (all instances, no quantization)
```bash
python run_qwen_evaluation.py \
    --wandb_project "swebench-qwen-full" \
    --wandb_run_name "qwen_oracle_lite_complete"
```

### Custom Dataset
```bash
python run_qwen_evaluation.py \
    --dataset_name "your-custom-dataset" \
    --wandb_project "swebench-custom"
```

## Results Interpretation

### Wandb Metrics

- **Pass Rate**: Percentage of instances that passed all tests
- **Instance Results**: Individual instance status and test results
- **Error Analysis**: Breakdown of failure types
- **Performance Metrics**: Timing and resource usage

### Local Files

- **`predictions_qwen.jsonl`**: Model predictions in SWE-bench format
- **`evaluation_results/{run_id}_report.json`**: Detailed evaluation results
- **`logs/`**: Detailed logs for debugging

## Next Steps

1. **Analyze Results**: Review wandb dashboard and local reports
2. **Iterate**: Modify model parameters or prompts based on results
3. **Scale**: Run on larger datasets or with different models
4. **Compare**: Run multiple experiments and compare results

## Support

For issues:
1. Check the troubleshooting section above
2. Review logs in the `logs/` directory
3. Check wandb dashboard for detailed metrics
4. Verify all prerequisites are installed correctly
