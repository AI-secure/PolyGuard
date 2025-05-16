import argparse
import json
import os
from sklearn.metrics import accuracy_score, f1_score
from guardrail_model import get_guardrail
# from jailbreak import jailbreak
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-Guard-3-8B", help="model_id")
    parser.add_argument("--jb_strategy", type=int, default=0, help="jailbreak strategy")
    args = parser.parse_args()

    print(args.model_id)
    # model = get_guardrail(args.model_id)
    # model = get_guardrail(args.model_id, cache_dir='/home/mintong/.cache/huggingface/hub/')
    # model = get_guardrail(args.model_id, cache_dir='/data1/common/hf_cache/')
    model = get_guardrail(args.model_id)

    policy_list = ["Google", "Microsoft", 'Amazon', 'Apple', 'Meta', 'NVIDIA', 'IBM', 'Intel', 'Adobe', 'ByteDance']
    policy_list += ['COLLEGE_BOARD_AP', 'CSU', 'AAMC', 'AI_FOR_EDUCATION', 'MCGOVERN_MED', 'NIU', 'SAMPLE_AI_GUIDANCE_TOOLKIT', 'UNESCO', 'IB']
    dataset_file_format = 'results/{}/data_{}.jsonl'
    model_output_dir = 'outputs/{}'

    for policy_name in policy_list:
        unsafe_labels = []
        dataset = []
        label_to_category = {'safe': 0, 'unsafe': 1}

        # Load data
        for label, category in label_to_category.items():
            with open(dataset_file_format.format(policy_name, label), 'r') as f:
                lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                data['label'] = label
                dataset.append(data)
                # prompts.append(data['instance'])
                unsafe_labels.append(category)

        prediction_dir = model_output_dir.format(policy_name)
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_file = os.path.join(prediction_dir, '{}.jsonl'.format(args.model_id.split('/')[-1]))

        predictions = []
        if os.path.exists(prediction_file):
            print(f"Found existing prediction file for {policy_name}. Loading predictions from {prediction_file}")
            # Load predictions from existing file
            with open(prediction_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["prediction"])
        else:
            # Run predictions
            print(f"No prediction file found for {policy_name}. Running model inference...")
            with open(prediction_file, 'w') as f:
                for data in tqdm(dataset, desc=f"Evaluating {policy_name}"):
                    result = model.moderate(data['instance'])
                    pred = 1 if result.flagged else 0
                    predictions.append(pred)
                    data['prediction'] = pred
                    f.write(json.dumps(data) + "\n")

            print(f"Saved predictions to {prediction_file}")

        # Calculate and print metrics
        accuracy = accuracy_score(unsafe_labels, predictions)
        f1 = f1_score(unsafe_labels, predictions)
        print(f"Dataset: {policy_name}, Accuracy: {accuracy*100:.2f}, F1: {f1*100:.2f}")
