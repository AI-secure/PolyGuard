from abc import ABC, abstractmethod
from typing import List

class BaseGuardrailModel(ABC):
    @abstractmethod
    def moderate(self, prompt: str or List[dict]) -> dict:
        pass

    # @abstractmethod
    # def batch_moderate(self, prompts: List[str] or List[List[dict]]) -> List[dict]:
    #     pass
