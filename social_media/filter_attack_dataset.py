import json
import os

if __name__=="__main__":
    domain = "Discord"

    model_ids = ["MD-Judge-v0_2-internlm2_7b", "wildguard", "Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0", "granite-guardian-3.2-5b", "llmjudge"]
    data_all = []
    for model_id in model_ids:
        file_path = f'./results/{domain}/{model_id}_all_records.jsonl'
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        data_all.append(data)

    filtered_data = []
    for i in range(len(data_all[0])):
        if data_all[0][i]["attribute"]=="unsafe":
            if all(data_all[k][i]["moderation_result"]==True for k in range(len(model_ids))):
                filtered_data.append(data_all[0][i])

    os.makedirs("./data_attack", exist_ok=True)
    with open(f"./data_attack/{domain}.jsonl", "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")