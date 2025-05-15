import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseGuardrailModel
from .utils import ModerationResult
from typing import List

class IBMGuard(BaseGuardrailModel):
    def __init__(self, model_id = "ibm-granite/granite-guardian-3.2-3b-a800m", device = "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
         # cache_dir = "/data1/common/hf_cache"
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        self.safe_token = "No"
        self.unsafe_token = "Yes"
        self.nlogprobs = 20

    def moderate(self, prompt: str or List[dict]) -> dict:

        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]

        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        guardian_config = {"risk_name": "harm"}
        input_ids = self.tokenizer.apply_chat_template(
            prompt, guardian_config=guardian_config, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )


        label, prob_of_risk = self.parse_output(output, input_len)

        if label == self.unsafe_token:
            moderation_result.flagged = True
        else:
            moderation_result.flagged = False
        moderation_result.reason = f"Prob of risk: {prob_of_risk}"
        return moderation_result

    def parse_output(self, output, input_len):
        label = None
        full_res = self.tokenizer.decode(output.sequences[:, input_len + 1:][0], skip_special_tokens=True).strip()
        confidence_level = full_res.removeprefix('<confidence>').removesuffix('</confidence>').strip()
        res = self.tokenizer.decode(output.sequences[:, input_len:input_len + 1][0], skip_special_tokens=True).strip()
        if self.unsafe_token.lower() == res.lower():
            label = self.unsafe_token
        elif self.safe_token.lower() == res.lower():
            label = self.safe_token
        else:
            label = "Failed"
        return label, confidence_level