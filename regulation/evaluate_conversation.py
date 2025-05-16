#!/usr/bin/env python3
import json
import os, sys
sys.path.append("./")
import argparse
import time
from tqdm import tqdm
import random
from guardrail_model import get_guardrail

def evaluate_guardrails_conversation():
    """Evaluate guardrail models on conversation prompts (safe and unsafe) and save results."""
    parser = argparse.ArgumentParser(description="Evaluate guardrail models on GuardBench conversation dataset")
    parser.add_argument("--model_id", "-m", type=str, required=True, help="Guardrail model ID to evaluate")
    parser.add_argument("--checkpoint", "-c", type=str, required=False, help="Checkpoint to evaluate")
    args = parser.parse_args()

    # Paths
    base_dir = "./"
    dataset_dir = f"{base_dir}/dataset/regulation/eu_ai_act/conversation"
    output_base_dir = f"{base_dir}/results/eu_ai_act"
    
    # Input files
    safe_file = f"{dataset_dir}/safe.jsonl"
    unsafe_file = f"{dataset_dir}/unsafe.jsonl"
    
    # Generate output directory based on model ID
    model_name = args.model_id.split("/")[-1]
    output_dir = os.path.join(output_base_dir, model_name, "conversation")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    safe_output_file = os.path.join(output_dir, "safe_result.jsonl")
    unsafe_output_file = os.path.join(output_dir, "unsafe_aug_result.jsonl")
    
    # Initialize the guardrail model
    print(f"Initializing guardrail model: {args.model_id}")
    model = get_guardrail(args.model_id, cache_dir=args.checkpoint)
    
    # # Process safe prompts
    # process_file(safe_file, safe_output_file, model, "safe")
    
    # Process unsafe prompts
    process_file(unsafe_file, unsafe_output_file, model, "unsafe")
    
    print(f"Evaluation complete. Results saved to {output_dir}")

def process_file(input_file, output_file, model, true_label):
    """Process a single JSONL file and save results."""
    # Load the prompts
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                prompts.append(json.loads(line))
    
    print(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Evaluate each prompt and write results
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for prompt_data in tqdm(prompts, desc=f"Evaluating {true_label} prompts"):
            user_prompt = prompt_data["prompt"]
            assistant_response = prompt_data["response"]
            
            # Format as a message with user and assistant roles
            formatted_message = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            
            try:
                # Get moderation result
                moderation_result = model.moderate(formatted_message)
                
                # Convert ModerationResult to dictionary
                moderation_dict = moderation_result.to_dict()
                
                # Extract flagged value
                is_flagged = moderation_dict["flagged"]
                
                # Create result entry
                result = {
                    "category": prompt_data.get("category", "unknown"),
                    "prompt": user_prompt,
                    "response": assistant_response,
                    "true_label": true_label,
                    "predicted_label": "unsafe" if is_flagged else "safe",
                    "flagged": is_flagged,
                }
                
                # Add original fields from the prompt data except "prompt" and "response" which are already included
                original_data = {k: v for k, v in prompt_data.items() if k not in ["prompt", "response"]}
                result.update(original_data)
                
                # Write the result as a JSON line
                out_f.write(json.dumps(result) + '\n')
                
                # Add a small random delay to avoid rate limits if any
                time.sleep(random.uniform(0.1, 0.3))
                
            except Exception as e:
                print(f"Error evaluating conversation '{user_prompt[:50]}...': {e}")
                continue

if __name__ == "__main__":
    evaluate_guardrails_conversation() 