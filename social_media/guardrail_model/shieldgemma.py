from .base import BaseGuardrailModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import List
from .utils import ModerationResult

class SheildGemma(BaseGuardrailModel):
    def __init__(self, model_id = "google/shieldgemma-2b", device = "cuda"):
        dtype = torch.bfloat16
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    def moderate(self, prompt: str or List[dict]) -> dict:
        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        if type(prompt) == str:
            # Assume the prompt is user message by default
            prompt = [{"role": "user", "content": prompt}]

        guideline = '''"No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).\n"No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).\n	"No Hate Speech": The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.\n"No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.'''
        inputs = self.tokenizer.apply_chat_template(prompt, guideline=guideline, return_tensors="pt", return_dict=True).to(self.model.device)

        output = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=100, pad_token_id=0, return_dict_in_generate=True, output_scores=True)
        sequence, scores = output.sequences, output.scores
        result = self.tokenizer.decode(sequence[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        if "yes" in result.lower():
            moderation_result.flagged = True
        else:
            moderation_result.flagged = False
        return moderation_result