import os
import json
import time

from .base import BaseGuardrailModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import List
from .utils import ModerationResult
import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests


class AzureModAPI(BaseGuardrailModel):
    def __init__(self, model_id = "azure", device = "cuda"):
        self.model_id = model_id
        self.endpoint = ""
        self.api_key = ""
        self.url = f"{self.endpoint}/contentsafety/text:analyze?api-version=2024-09-15-preview"
        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

    def moderate(self, prompt: str or List[dict]) -> dict:
        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        if type(prompt) != str:
            str_prompt = ""
            for p in prompt:
                role = p["role"]
                content = p["content"]
                str_prompt += f"{role}: {content}\n"
            prompt = str_prompt
        if prompt == "":
            prompt = "NULL"

        data = {
            "text": prompt,
        }


        response = requests.post(self.url, headers=self.headers, json=data)
        result = response.json()
        is_unsafe = any(item['severity'] >= 3 for item in result['categoriesAnalysis'])
        moderation_result.flagged = is_unsafe

        return moderation_result