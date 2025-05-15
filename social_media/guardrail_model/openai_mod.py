from .base import BaseGuardrailModel
from openai import OpenAI
import json
from tqdm import tqdm
from typing import List
from .utils import ModerationResult


class OpenAIMod(BaseGuardrailModel):
    def __init__(self, model_id = "omni-moderation-latest", device = "cpu"):
        self.client = OpenAI()
        self.model_id = model_id

    def moderate(self, prompt: str or List[dict]) -> dict:

        if type(prompt) != str:
            str_prompt = ""
            for p in prompt:
                role = p["role"]
                content = p["content"]
                str_prompt += f"{role}: {content}\n"
            prompt = str_prompt

        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        response = self.client.moderations.create(model=self.model_id, input=prompt)
        results = response.results
        category_scores = results[0].category_scores
        moderation_result.flagged = results[0].flagged
        moderation_result.score = category_scores
        return moderation_result