from abc import abstractmethod
from typing import Any, Optional
import logging

import backoff
import google.api_core
import google.api_core.exceptions
import google.genai.errors
import pandas as pd
import google.genai
import mistralai
import google

from src.models._base import BaseModel
from src.data.utils import jsonize_rows
from src.models.config import MISTRAL_API_KEY, GEMINI_API_KEY, BACKOFF_MAX_TRIES, BACKOFF_FACTOR
from src.models.utils import log_backoff

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
        class_column: str = "Label"
    ) -> list[str]:
        """
        Classifies test input data using an LLM, ensuring the output is one of the possible labels.

        Args:
            icl_data: A pandas DataFrame containing the ICL examples.
            test_data: A pandas DataFrame containing the test data to be classified.
            task: A sentence that further explains the classification task.
            class_column: The name of the column containing the labels in `icl_data`.

        Returns:
            A list containing the predicted labels for each test input.
        """
        
        # Extract possible labels from icl_data
        possible_labels = set(icl_data[class_column].unique())
        
        # Prepare few-shot context
        icl_inputs = jsonize_rows(icl_data.drop(columns=[class_column]))
        context = (
            f"You are performing the following classification task: {task}\n\n"
            "Here are some examples:\n"
        )
        for i, o in zip(icl_inputs, icl_data[class_column]):
            context += str({"Input": i, "Output": o}) + "\n"
        context += "Where Output is the classification label for each data entry."
        
        # Define instructions
        pre_instruction = "Classify the following input and provide the corresponding Output:\n"
        last_instruction = (
            # f"Your answer must be exactly one of the following labels: {', '.join(possible_labels)}.\n"
            "Ensure you provide only the raw Output and nothing else."
        )
        
        results = []
        logging.info('Starting predictions...asking the model...')        
        test_inputs = jsonize_rows(test_data)
        for i, test_input in enumerate(test_inputs):
            logging.info(f'Prediction nº{i+1}...')
            instructions = pre_instruction + test_input + "\n" + last_instruction
            response = self.ask(instructions=instructions, context=context)
            response_text = response["answer"].strip()
        
            # Look for a valid label in the response
            model_output = None
            for label in possible_labels:
                if label in response_text:
                    model_output = label
                    break
            
            if model_output is None:
                logging.warning(f'No valid label found in response: {response_text}')
                model_output = None
            
            logging.info(f'Response: {model_output}')
            results.append(model_output)
        
        return results

class Mistral(LLModel):
    """
    Mistral assitant API following https://docs.mistral.ai/api/
    """

    def __init__(self, model: str = "mistral-large-latest"):

        super().__init__(model)
        self.client = mistralai.Mistral(api_key=MISTRAL_API_KEY)

    @backoff.on_exception(
        backoff.expo,
        mistralai.models.SDKError,
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=backoff.full_jitter,
        giveup=lambda e: "429" not in str(e),  # Retry only on 429 errors
        on_backoff=log_backoff
    )
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

    @backoff.on_exception(
        backoff.expo,
        exception=google.genai.errors.ClientError,
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=backoff.full_jitter,
        giveup=lambda e: "429" not in str(e),
        on_backoff=log_backoff
    )
    def ask(self, instructions: str, context: Optional[str] = None) -> dict[str, str]:


        got_response = False
        while not got_response:

            context = context if context else ""
            response = self.client.models.generate_content(
                model=self.model, contents=context + "\n" + instructions
            )

            try:
                response_dict = {
                    "answer": response.text,
                    "finish_reason": response.candidates[0].finish_reason.name
                }
                got_response = True
            
            except Exception:

                got_response = False

        return response_dict

        
