import json
import os
from tqdm import tqdm
from guardrail_model import get_guardrail

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Run eval")

parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="input_file"
)
parser.add_argument(
    "--prompt_or_chat",
    type=str,
    required=True,
    help="prompt_or_chat"
)


args = parser.parse_args()
input_file = args.input_file
dataset_name = input_file.split("/")[-1].split(".")[0]
prompt_or_chat = args.prompt_or_chat

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    # model_ids = ["meta-llama/Llama-Guard-3-8B", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/Llama-Guard-3-1B", "meta-llama/LlamaGuard-7b"]
    # model_ids += ["meta-llama/Llama-Guard-4-12B"]
    # model_ids += ["google/shieldgemma-2b", "google/shieldgemma-9b"]
    # model_ids += ["text-moderation-latest", "omni-moderation-latest"]
    model_ids = ["OpenSafetyLab/MD-Judge-v0_2-internlm2_7b", "OpenSafetyLab/MD-Judge-v0.1"]
    model_ids += ["allenai/wildguard"]
    model_ids += ["nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0", "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"]
    model_ids += ["ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-granite/granite-guardian-3.2-5b"]
    model_ids += ["llmjudge_gpt_4o"]
    model_ids += ["azure"]
    model_ids += ["aws"]

    cache_dir = '/data1/common/hf_cache'
    # cache_dir = '/home/chengquan/.cache/huggingface/hub'
    
    result_dir = "/home/chengquan/GuardBench/result/" + input_file.split("/")[-2]
    # result_dir = "/home/chengquan/GuardBench/result"
    os.makedirs(result_dir, exist_ok=True)


    for model_id in model_ids:
        # sleep 5 秒
        import time
        time.sleep(5)
        if "allenai" in model_id or "OpenSafetyLab" in model_id or "ibm" in model_id or "Aegis" in model_id:
            cache_dir = None
            
        output_filename = f"{model_id.replace('/', '_')}_{dataset_name}_{prompt_or_chat}_moderation_result.json"
        output_path = os.path.join(result_dir, output_filename)
        if os.path.exists(output_path):
            print(f"This of {output_filename} result already exist, skip...")
            continue
        print(f"\n>>> Loading model: {model_id}")
        try:
            model = get_guardrail(model_id, cache_dir=cache_dir)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            continue

        
        print(f">>> Evaluating dataset: {dataset_name}")
        try:
            data = load_json(input_file)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        for item in tqdm(data, desc=f"[{model_id}] {dataset_name}"):
            prompt = item.get("gen_prompt", "")
            response = item.get("gen_response", "")
            if not prompt:
                continue
            chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response},]
            
            try:
                if(prompt_or_chat=="prompt"):
                    moderation_result = model.moderate(prompt)
                    output_filename = f"{model_id.replace('/', '_')}_{dataset_name}_prompt_moderation_result.json"
                elif(prompt_or_chat == "chat"):
                    moderation_result = model.moderate(chat)
                    output_filename = f"{model_id.replace('/', '_')}_{dataset_name}_chat_moderation_result.json"
                result_key = f"moderation_result"
                item[result_key] = str(moderation_result)
                print("moderation_result: ",str(moderation_result))
            except Exception as e:
                item[result_key] = f"[Error] {e}"
        
        output_path = os.path.join(result_dir, output_filename)
        save_json(output_path, data)
        print(f"Saved moderation results to: {output_path}")
