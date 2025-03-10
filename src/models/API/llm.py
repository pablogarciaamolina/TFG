from abc import abstractmethod
from typing import Any, Optional
import logging

import pandas as pd
import google.genai
import mistralai
import google

from src.models._base import BaseModel
from src.data.utils import jsonize_rows
from src.models.config import MISTRAL_API_KEY, GEMINI_API_KEY

class LLModel(BaseModel):
    
    client: Any

    def __init__(self, model: str):
        
        super().__init__(model)
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

    def predict(self,
        icl_data: pd.DataFrame,
        test_data: pd.DataFrame,
        task: str = "Classify data",
        class_column: str = "Label",
    ) -> list[str]:
        """
        Method for classifying test input data using an LLM

        Args:
            icl_data: A pandas Dataframe containing the ICL examples
            test_data: A pandas Dataframe containing the test data to be clasified
            task: A sentence that further explains the classification task
            class_column: The name of the column containing the labels in the ``icl_data``

        Return:
            A list containing the labels for each of the test inputs in order.
        """
        
        icl_inputs = jsonize_rows(icl_data.drop(columns=["Label"]))
        context = (
            f"You are performing the following classification task: {task}\n\n"
            "Here are some examples:\n"
        )
        for i, o in zip(icl_inputs, icl_data[class_column]):
            context += str(
                {
                    "Input": i,
                    "Output": o
                }
            ) + "\n"
        context += f"Where Output is the classification label for each data entry."
        
        test_inputs = jsonize_rows(test_data)
        pre_instruction = f"Classify the following input and provide the corresponding Output:\n"
        last_instruction = f"Your answer must only be the Output for the Input. Make sure you provide the raw Output, and nothing else."
        
        results = []
        logging.info('Starting predictions...asking the model...')
        for i in range(len(test_inputs)):
            logging.info(f'Prediction nº{i+1}...')
            instructions = pre_instruction + test_inputs[i] + "\n" + last_instruction
            response = self.ask(instructions=instructions, context=context)
            logging.info(f'Response: {response["answer"]}')
            results.append(response["answer"])
        
        return results


class Mistral(LLModel):
    """
    Mistral assitant API following https://docs.mistral.ai/api/
    """

    def __init__(self, model: str = "mistral-large-latest"):

        super().__init__(model)
        self.client = mistralai.Mistral(api_key=MISTRAL_API_KEY)

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
    
class Gemini(LLModel):

    def __init__(self, model: str = "gemini-2.0-pro-exp-02-05"):
        
        super().__init__(model)
        self.client = google.genai.Client(api_key=GEMINI_API_KEY)

    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:

        context = context if context else ""
        response = self.client.models.generate_content(
            model=self.model, contents=context + "\n" + instructions
        )

        return {
            "answer": response.text,
            "finish_reason": response.candidates[0].finish_reason.name
        }
        
