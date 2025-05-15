from .base import BaseGuardrailModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import List
from .utils import ModerationResult
# from transformers import AutoProcessor, Llama4ForConditionalGeneration

class LlamaGuard(BaseGuardrailModel):
    def __init__(self, model_id = "meta-llama/Llama-Guard-3-8B", device = "cuda"):
        dtype = torch.bfloat16
        self.model_id = model_id
        self.device = device
        if self.model_id == "meta-llama/Llama-Guard-4-12B":
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

        if model_id == "meta-llama/Llama-Guard-3-8B" or model_id == "meta-llama/Llama-Guard-4-12B":
            self.categories = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14']
        elif model_id == "meta-llama/Meta-Llama-Guard-2-8B":
            self.categories = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        elif model_id == "meta-llama/Llama-Guard-3-1B":
            self.categories = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'] # support adding or excluding categories
        elif model_id == "meta-llama/LlamaGuard-7b":
            self.categories = ["0", "1", "2", "3", "4", "5", "6"]
        else:
            raise ValueError(f"Unsupported model_id {model_id}")
        # self.category_token_ids = [int(self.tokenizer.encode(cat, return_tensors="pt").to(device)[0, -1]) for cat in self.categories]

    def moderate(self, prompt: str or List[dict]) -> dict:
        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        if type(prompt) == str and self.model_id != "meta-llama/Llama-Guard-4-12B":
            # Assume the prompt is user message by default
            prompt = [{"role": "user", "content": prompt}]
        elif type(prompt) == str and self.model_id == "meta-llama/Llama-Guard-4-12B":
            prompt = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if self.model_id != "meta-llama/Llama-Guard-4-12B":
            input_ids = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0, return_dict_in_generate=True, output_scores=True)
            sequence, scores = output.sequences, output.scores
            result = self.tokenizer.decode(sequence[0][input_ids.shape[-1]:], skip_special_tokens=False)
        else:
            inputs = self.processor.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
            result = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        if "unsafe" in result:
            moderation_result.flagged = True
            moderation_result.reason = result.strip().split("\n")[-1]
        else:
            moderation_result.flagged = False

        # determine category-wise likelihoods if flagged
        # output_ids = self.tokenizer.encode(result, return_tensors="pt")[0]
        # token_id_safe = int(self.tokenizer.encode("safe", return_tensors="pt").to("cuda")[0, -1])
        # token_id_unsafe = int(self.tokenizer.encode("unsafe", return_tensors="pt").to("cuda")[0, -1])
        # for output_id, score in zip(output_ids, scores):
        #     if output_id == token_id_unsafe or output_id == token_id_safe:
        #         probs_ = torch.softmax(score[0], dim=-1)
        #         prob_safe = float(probs_[token_id_safe])
        #         prob_unsafe = float(probs_[token_id_unsafe])
        #         prob_unsafe /= (prob_safe + prob_unsafe)
        #         moderation_result.prob_flagged = prob_unsafe
        #     elif output_id in self.category_token_ids:
        #         probs_ = torch.softmax(score[0], dim=-1)
        #         probs_all = [float(probs_[tok_id]) for tok_id in self.category_token_ids]
        #         normalization_fac = sum(probs_all)
        #         category_scores = [p / normalization_fac for p in probs_all]
        #         moderation_result.score = category_scores
        return moderation_result