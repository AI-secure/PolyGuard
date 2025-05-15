from .llamaguard import LlamaGuard
from .shieldgemma import SheildGemma
from .openai_mod import OpenAIMod
from .mdjudge import MDJudge
from .wildguard import WildGuard
from .llmjudge import LLMJudge
from .aegis import Aegis
from .ibm_guard import IBMGuard
from .azure import AzureModAPI
from .aws_bedrock import AWSModAPI

def get_guardrail(model, device="gpu"):
    if "llama" in model.lower() and "aegis" not in model.lower():
        return LlamaGuard(model, device)
    elif "gemma" in model.lower():
        return SheildGemma(model, device)
    elif "moderation" in model.lower():
        return OpenAIMod(model, device)
    elif "llmjudge" in model.lower():
        return LLMJudge(model, device)
    elif "judge" in model.lower():
        return MDJudge(model, device)
    elif "wildguard" in model.lower():
        return WildGuard(model, device)
    elif "aegis" in model.lower():
        return Aegis(model, device)
    elif "ibm" in model.lower():
        return IBMGuard(model, device)
    elif "azure" in model.lower():
        return AzureModAPI(model, device)
    elif "aws" in model.lower():
        return AWSModAPI(model, device)
    else:
        raise ValueError(f"Unsupported guardrail model {model}")
