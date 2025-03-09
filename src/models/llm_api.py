from abc import ABC, abstractmethod
from typing import Any, Optional

from mistralai import Mistral
from google import genai

from .config import MISTRAL_API_KEY, GEMINI_API_KEY

class BaseChat(ABC):
    
    client: Any

    def __init__(self, model: str):
        
        self.model = model

    @abstractmethod
    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:
        """
        Interface for chat completion task

        Args:
            instructions: String containing the questions for the model
            context: Optional string containing behavioural context for the model, in order to indicate which role or position to adopt

        Returns:
            :A dictionary containing the reply from the model as well as the exit reason of the model (control purposes). Such like:
                {
                    "answer": <str contining the answer from the model>,
                    "finish_reason": <exit reason in str format>
                }
        """

        pass


class MistralChat(BaseChat):
    """
    Mistral assitant API following https://docs.mistral.ai/api/
    """

    def __init__(self, model: str = "mistral-large-latest"):

        super().__init__(model=model)
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:

        messages: list[dict] = [{
            "role": "system",
            "content": context
        }] if context else []
        messages.append({
            "role": "user",
            "content": instructions
        })

        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )

        return {
            "answer": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason
        }
    
class GeminiChat(BaseChat):

    def __init__(self, model: str = "gemini-2.0-pro-exp-02-05"):
        
        super().__init__(model=model)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:

        response = self.client.models.generate_content(
            model=self.model, contents=context + "\n" + instructions
        )

        return {
            "answer": response.text,
            "finish_reason": response.candidates[0].finish_reason.name
        }
        
