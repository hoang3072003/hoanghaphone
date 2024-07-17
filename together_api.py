import os
from litellm import completion
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

set_verbose=True
class TogetherLLM:
    def __init__(self,
                 model: str = "together_ai/meta-llama/Llama-3-70b-chat-hf",
                 together_api_key: str = "655baa18d5f594c3031513b9cf40b3b81f6b74d24b4e05baa0e43fcaef2bc650",
                 temperature: float = 0.7,
                 max_tokens: int = 512):
        self.model = model
        self.together_api_key = together_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(
        self,
        messages: list,
        stream=True
    ):
        """Call to Together."""
        output = completion(
            messages=messages,
            model="together_ai/meta-llama/Llama-3-70b-chat-hf",
            together_api_key=self.together_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream
        )

        if stream:
            return output
        else:
            return output['choices'][0]['message']['content']
