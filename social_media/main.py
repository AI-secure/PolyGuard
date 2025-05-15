import os

from openai import OpenAI
import json
from tqdm import tqdm
from guardrail_model import get_guardrail
import argparse

def update_final_result(res, final_result):
    if res["category"] not in final_result.keys():
        final_result[res["category"]] = {}
        final_result[res["category"]]["size_test"] = 0
        final_result[res["category"]]["size_flagged"] = 0
    if res["rule"] not in final_result[res["category"]].keys():
        final_result[res["category"]][res["rule"]] = {}
        final_result[res["category"]][res["rule"]]["size_test"] = 0
        final_result[res["category"]][res["rule"]]["size_flagged"] = 0
    final_result[res["category"]]["size_test"] += 1
    final_result[res["category"]]["size_flagged"] += int(res["moderation_result"].flagged)
    final_result[res["category"]][res["rule"]]["size_test"] += 1
    final_result[res["category"]][res["rule"]]["size_flagged"] += int(res["moderation_result"].flagged)

def summarize_result(results, save_dir, model_id):

    final_result_unsafe, final_result_safe = {}, {}
    for res in results:
        if res["attribute"] == "unsafe":
            update_final_result(res, final_result_unsafe)
        elif res["attribute"] == "safe":
            update_final_result(res, final_result_safe)

    tol = 1e-8

    precision, recall, f1 = {}, {}, {}
    for cat in final_result_unsafe.keys():
        precision[cat] = {}
        recall[cat] = {}
        f1[cat] = {}
    for cat in final_result_unsafe.keys():
        TP = final_result_unsafe[cat]["size_flagged"]
        FP = final_result_safe[cat]["size_flagged"]
        FN = final_result_unsafe[cat]["size_test"] - final_result_unsafe[cat]["size_flagged"]
        precision[cat]["overall"] = 1.0 * TP / max((TP+FP), tol)
        recall[cat]["overall"] = 1.0 * TP / max(TP+FN, tol)
        f1[cat]["overall"] = 2.0 * precision[cat]["overall"] * recall[cat]["overall"] / max(precision[cat]["overall"] + recall[cat]["overall"], tol)
        for rule in final_result_unsafe[cat].keys():
            if rule != "size_test" and rule != "size_flagged":
                TP = final_result_unsafe[cat][rule]["size_flagged"]
                FP = final_result_safe[cat][rule]["size_flagged"]
                FN = final_result_unsafe[cat][rule]["size_test"] - final_result_unsafe[cat][rule]["size_flagged"]
                precision[cat][rule] =  1.0 * TP / max((TP+FP), tol)
                recall[cat][rule] = 1.0 * TP / max(TP+FN, tol)
                f1[cat][rule] = 2.0 * precision[cat][rule] * recall[cat][rule] / max(precision[cat][rule] + recall[cat][rule], tol)
    precision["average"], recall["average"], f1["average"] = 0., 0., 0.
    for cat in final_result_unsafe.keys():
        precision["average"] += precision[cat]["overall"] / len(final_result_unsafe.keys())
        recall["average"] += recall[cat]["overall"] / len(final_result_unsafe.keys())
        f1["average"] += f1[cat]["overall"] / len(final_result_unsafe.keys())
    with open(os.path.join(save_dir, f"{model_id}_precision.json"), "w", encoding="utf-8") as f:
        json.dump(precision, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, f"{model_id}_recall.json"), "w", encoding="utf-8") as f:
        json.dump(recall, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, f"{model_id}_f1.json"), "w", encoding="utf-8") as f:
        json.dump(f1, f, indent=2, ensure_ascii=False)


def run_instance(instance, model, attribute):
    prompt = instance["instance"]
    safety_category = instance["category"]
    safety_rule = instance["rule"]
    moderation_result = model.moderate(prompt)
    record = {"instance": prompt, "category": safety_category, "rule": safety_rule, "moderation_result": moderation_result, "attribute": attribute}
    return record

if __name__ == "__main__":

    model_ids = ["meta-llama/Llama-Guard-4-12B", "meta-llama/Llama-Guard-3-8B", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/Llama-Guard-3-1B",
                 "meta-llama/LlamaGuard-7b"]
    model_ids += ["google/shieldgemma-2b", "google/shieldgemma-9b"]
    model_ids += ["text-moderation-latest", "omni-moderation-latest"]
    model_ids += ["OpenSafetyLab/MD-Judge-v0_2-internlm2_7b", "OpenSafetyLab/MD-Judge-v0.1"]
    model_ids += ["allenai/wildguard"]
    model_ids += ["nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0",
                  "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"]
    model_ids += ["ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-granite/granite-guardian-3.2-5b"]
    model_ids += ["llmjudge"]
    model_ids += ["azure"]
    model_ids += ["aws"]


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=model_ids)
    parser.add_argument('--domain', type=str, choices=["Reddit","Twitter","Instagram","Discord","Youtube","Sportify"])
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    data_unsafe, data_safe = [], []
    with open(f'./datagen/results/{args.domain}/data_unsafe.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            data_unsafe.append(json.loads(line.strip()))
    with open(f'./datagen/results/{args.domain}/data_safe.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            data_safe.append(json.loads(line.strip()))


    model = get_guardrail(args.model, args.device)

    save_dir = f"./results/{args.domain}"
    os.makedirs(save_dir, exist_ok=True)


    results = []
    for instance_unsafe, instance_safe in tqdm(zip(data_unsafe, data_safe)):
        record_unsafe = run_instance(instance_unsafe, model, "unsafe")
        record_safe = run_instance(instance_safe, model, "safe")
        results.append(record_unsafe)
        results.append(record_safe)
        if len(results) % 100 == 0:
            summarize_result(results, save_dir=save_dir, model_id=args.model.split('/')[-1])

    summarize_result(results, save_dir=save_dir, model_id=args.model.split('/')[-1])

    with open(os.path.join(save_dir, f"{args.model.split('/')[-1]}_all_records.jsonl"), "w", encoding="utf-8") as f:
        for entry in results:
            entry["moderation_result"] = entry["moderation_result"].flagged
            json.dump(entry, f)
            f.write("\n")