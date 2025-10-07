"""
Qwen2.5-Coder-7B inference script for SWE-bench.

This module provides inference capabilities for the Qwen2.5-Coder-7B model
on SWE-bench tasks.
"""

import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import wandb

from swebench.harness.utils import load_swebench_dataset
from swebench.harness.wandb_logging import InferenceLogger
from swebench.harness.text_utils import remove_readme

logger = logging.getLogger(__name__)


class QwenInference:
    """Qwen2.5-Coder-7B inference class for SWE-bench."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        use_quantization: bool = False,
        quantization_config: Optional[Dict] = None,
        max_length: int = 8192,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        Initialize the Qwen inference model.
        
        Args:
            model_name: Name of the model to load
            device_map: Device mapping strategy
            torch_dtype: Torch data type for the model
            use_quantization: Whether to use quantization
            quantization_config: Quantization configuration
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
        """
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }
        
        # Add quantization config if specified
        if self.use_quantization and self.quantization_config:
            bnb_config = BitsAndBytesConfig(**self.quantization_config)
            model_kwargs["quantization_config"] = bnb_config
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        logger.info("Model loaded successfully")
    
    def generate_patch(
        self,
        problem_statement: str,
        repo_content: str,
        hints_text: str = "",
        max_new_tokens: int = 2048,
    ) -> tuple[str, dict]:
        """
        Generate a patch for the given problem statement.
        
        Args:
            problem_statement: The problem description
            repo_content: Repository content/context
            hints_text: Additional hints
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tuple of (patch, metadata) where metadata contains raw and cleaned completions
        """
        # Construct the prompt
        prompt = self._construct_prompt(problem_statement, repo_content, hints_text)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
            padding=True
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response (raw completion)
        raw_completion = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract patch from generated text (cleaned completion)
        patch = self._extract_patch(raw_completion)
        
        # Create metadata
        metadata = {
            "raw_completion": raw_completion,
            "cleaned_completion": patch,
            "prompt_length": len(prompt),
            "raw_completion_length": len(raw_completion),
            "patch_length": len(patch),
        }
        
        return patch, metadata
    
    def _construct_prompt(
        self,
        problem_statement: str,
        repo_content: str,
        hints_text: str = "",
    ) -> str:
        """Construct the prompt for the model."""
        
        # Remove README from problem statement to reduce noise
        problem_statement = remove_readme(problem_statement)
        
        prompt = f"""<|im_start|>system
You are an expert software engineer. Your task is to generate a patch (diff) that fixes the given problem. 

CRITICAL PATCH FORMAT RULES:
1. Output ONLY the patch - no explanations, no markdown, no code fences
2. Start immediately with "--- a/path/to/file.py"
3. Each hunk must have sufficient context (3+ lines before and after changes)
4. End each file's patch with a blank line
5. If patching multiple files, add a blank line between each file's patch
6. Include proper trailing newline at the end of the patch
7. Use exact context from the actual code - verify line numbers match

Patch Structure:
```
--- a/file1.py
+++ b/file1.py
@@ -10,5 +10,6 @@ function_name():
 context line 1
 context line 2
-old line
+new line
 context line 3
 context line 4

--- a/file2.py
+++ b/file2.py
@@ -20,3 +20,4 @@
...
```

<|im_end|>
<|im_start|>user
Problem Statement:
{problem_statement}

Repository Context:
{repo_content[:4000]}  # Truncate to avoid token limits

{f"Hints: {hints_text}" if hints_text else ""}

Generate ONLY a valid unified diff patch. Start with "---" immediately.

<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def _extract_patch(self, generated_text: str) -> str:
        """Extract the patch from the generated text."""
        # Remove markdown code fences if present
        cleaned_text = generated_text
        if '```' in cleaned_text:
            # Extract content between code fences
            parts = cleaned_text.split('```')
            for part in parts:
                if '---' in part and '+++' in part:
                    cleaned_text = part.strip()
                    # Remove language specifier if present
                    if cleaned_text.startswith(('diff', 'patch')):
                        cleaned_text = '\n'.join(cleaned_text.split('\n')[1:])
                    break
        
        # Look for diff markers
        lines = cleaned_text.split('\n')
        patch_lines = []
        in_patch = False
        blank_line_count = 0
        
        for i, line in enumerate(lines):
            # Start of a file patch
            if line.startswith('---'):
                in_patch = True
                blank_line_count = 0
                patch_lines.append(line)
            elif line.startswith('+++') and in_patch:
                patch_lines.append(line)
            elif in_patch:
                # Inside patch - include diff content
                if line.startswith('@@') or line.startswith('+') or line.startswith('-') or line.startswith(' ') or line.startswith('\\'):
                    patch_lines.append(line)
                    blank_line_count = 0
                elif line.strip() == '':
                    # Preserve blank lines (they might separate files or be part of context)
                    patch_lines.append(line)
                    blank_line_count += 1
                    
                    # If we have multiple consecutive blank lines and next line doesn't look like patch content
                    # this might be the end of the patch
                    if blank_line_count >= 2 and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if not (next_line.startswith(('---', '+++', '@@', '+', '-', ' ', '\\')) or next_line.strip() == ''):
                            break
                elif line.startswith(('diff --git', 'index ')):
                    # Git metadata - include it
                    patch_lines.append(line)
                    blank_line_count = 0
                else:
                    # Non-patch content
                    # Only stop if we've collected substantial patch content
                    if len(patch_lines) > 3 and any(l.startswith('@@') for l in patch_lines):
                        break
        
        result = '\n'.join(patch_lines)
        
        # Ensure trailing newline
        if result and not result.endswith('\n'):
            result += '\n'
        
        return result.strip()
    
    def run_inference_on_dataset(
        self,
        dataset_name: str,
        output_path: str,
        max_instances: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run inference on the entire dataset.
        
        Args:
            dataset_name: Name of the dataset
            output_path: Path to save predictions
            max_instances: Maximum number of instances to process
            instance_ids: Specific instance IDs to process
            
        Returns:
            List of predictions
        """
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_swebench_dataset(dataset_name)
        
        # Filter dataset if needed
        if instance_ids:
            dataset = [inst for inst in dataset if inst.get('instance_id', inst.get('id')) in instance_ids]
            logger.info(f"Filtered to {len(dataset)} instances")
        
        if max_instances:
            dataset = dataset[:max_instances]
            logger.info(f"Limited to {len(dataset)} instances")
        
        predictions = []
        
        for i, instance in enumerate(dataset):
            # Handle both dict and object formats
            if isinstance(instance, dict):
                instance_id = instance.get('instance_id', instance.get('id'))
                problem_statement = instance.get('problem_statement', instance.get('text', ''))
                repo_content = instance.get('repo_content', '')
                hints_text = instance.get('hints_text', '')
            else:
                instance_id = instance.instance_id
                problem_statement = instance.problem_statement
                repo_content = getattr(instance, 'repo_content', '')
                hints_text = getattr(instance, 'hints_text', '')
            
            logger.info(f"Processing instance {i+1}/{len(dataset)}: {instance_id}")
            
            try:
                # Generate patch
                patch, metadata = self.generate_patch(
                    problem_statement=problem_statement,
                    repo_content=repo_content,
                    hints_text=hints_text,
                )
                
                # Log raw and cleaned completions to console/file
                logger.info(f"[{instance_id}] Raw completion length: {metadata['raw_completion_length']} chars")
                logger.info(f"[{instance_id}] Cleaned patch length: {metadata['patch_length']} chars")
                logger.info(f"[{instance_id}] Raw completion:\n{metadata['raw_completion']}")
                logger.info(f"[{instance_id}] Cleaned patch:\n{metadata['cleaned_completion']}")
                
                # Check if patch is valid (has diff markers)
                has_diff_markers = '---' in patch or '+++' in patch or '@@' in patch
                
                prediction = {
                    "instance_id": instance_id,
                    "model_name_or_path": self.model_name,
                    "model_patch": patch,
                    "metadata": metadata,
                }
                
                predictions.append(prediction)
                
                # Log metrics to wandb (not full completions)
                if wandb.run is not None:
                    # Note: wandb logger is managed externally in main()
                    pass
                
            except Exception as e:
                logger.error(f"Error processing {instance_id}: {e}")
                # Add failed prediction
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": self.model_name,
                    "model_patch": "",
                })
        
        # Save predictions
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        logger.info(f"Predictions saved to: {output_path}")
        
        return predictions


def main():
    """Main function for running inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Coder-7B inference on SWE-bench")
    parser.add_argument("--dataset_name", type=str, default="ryankamiri/SWE-bench_oracle_lite")
    parser.add_argument("--output_path", type=str, default="predictions_qwen.jsonl")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--instance_ids", type=str, nargs="+", default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="swebench-qwen")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize wandb logger
    wandb_logger = InferenceLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"qwen-{args.dataset_name.split('/')[-1]}",
        config={
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "use_quantization": args.use_quantization,
            "max_instances": args.max_instances,
        }
    )
    
    # Setup quantization config if needed
    quantization_config = None
    if args.use_quantization:
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
    
    # Initialize inference
    inference = QwenInference(
        model_name=args.model_name,
        use_quantization=args.use_quantization,
        quantization_config=quantization_config,
    )
    
    # Run inference
    predictions = inference.run_inference_on_dataset(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        max_instances=args.max_instances,
        instance_ids=args.instance_ids,
    )
    
    # Log final results
    successful = len([p for p in predictions if p["model_patch"]])
    wandb_logger.log_final_inference(
        total_predictions=len(predictions),
        successful_predictions=successful
    )
    
    wandb_logger.finish()
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
