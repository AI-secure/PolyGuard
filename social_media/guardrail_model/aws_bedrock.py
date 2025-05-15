import os
import json
from .base import BaseGuardrailModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import List
from .utils import ModerationResult
import http.client, urllib.request, urllib.parse, urllib.error, base64
import boto3
import botocore.auth
import botocore.session
import requests
import json
from datetime import datetime

# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_DEFAULT_REGION=us-west-2

class AWSModAPI(BaseGuardrailModel):
    def __init__(self, model_id = "aws", device = "cuda"):
        self.model_id = model_id
        self.bedrock_client = boto3.client("bedrock-runtime",
                                      region_name="us-east-2",
                                      aws_access_key_id="",
                                      aws_secret_access_key="")


    def moderate(self, prompt: str or List[dict]) -> dict:
        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        if type(prompt) != str:
            str_prompt = ""
            for p in prompt:
                role = p["role"]
                content = p["content"]
                str_prompt += f"{role}: {content}\n"
            prompt = str_prompt


        response = self.bedrock_client.apply_guardrail(
            guardrailIdentifier="1bf70gdti5za",
            guardrailVersion="DRAFT",
            source='INPUT',
            content=[{
                'text': {
                    'text': prompt
                }
            }]
        )
        moderation_result.flagged = response['action'] != 'NONE'
        return moderation_result