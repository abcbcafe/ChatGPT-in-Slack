import abc
from abc import ABC

import anthropic


class LlmAdapter(ABC):
    @abc.abstractmethod
    def generate_text(self, messages):
        pass


class AnthropicAdapter(LlmAdapter):
    def __init__(self,
                 api_key: str,
                 system_prompt: str = "You are the Allied Mastercomputer. Be succinct.",
                 model: str = "claude-3-sonnet-20240229",
                 temperature: float = 0.95
                 ):
        self.temperature = temperature
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt

    def generate_text(self, messages):
        print(">>> Calling Anthropic")

        completion = anthropic.Anthropic().messages.create(
            # model="claude-3-opus-20240229",
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"{messages}"}
            ]
        )
        text = completion.content[0].text
        print(text)
        return text
