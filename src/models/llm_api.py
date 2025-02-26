from .config import MISTRAL_API_KEY, GEMINI_API_KEY
from mistralai import Mistral
from google import genai


class MistralChat:
    """
    Mistral assitant API following https://docs.mistral.ai/api/
    """

    def __init__(self, model: str = "mistral-large-latest"):

        self.model=model
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def ask(self,  instruction: str, context: str|None = None) -> dict[str, str]:
        """
        Interface for chat completion task

        Args:
            instruction: String containing the question for the model
            context: String containing behavioural context for the model, in order to indicate which role or position to adopt
        """

        messages: list[dict] = [{
            "role": "system",
            "content": context
        }] if context else []
        messages.append({
            "role": "user",
            "content": instruction
        })

        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )

        return {
            "answer": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason
        }
    
class GeminiChat:

    def __init__(self, model: str = "gemini-2.0-pro-exp-02-05"):
        
        self.model = model
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def ask(self, instruction: str) -> dict[str, str]:
        """
        Interface for chat completion task

        Args:
            instruction: String containing the question for the model
        """

        response = self.client.models.generate_content(
            model=self.model, contents=instruction
        )

        return {
            "answer": response.text,
            "finish_reason": response.candidates[0].finish_reason.name
        }
        
