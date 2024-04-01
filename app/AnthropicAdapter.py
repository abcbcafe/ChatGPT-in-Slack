import enum
import os
import re

import anthropic
from dotenv import load_dotenv
import requests

from app.LlmAdapter import LlmAdapter


class AnthropicModelType(enum.Enum):
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    # Add more model types as needed


class AnthropicAdapter(LlmAdapter):

    def __init__(
        self,
        api_key: str,
        system_prompt: str = "You are the Allied Mastercomputer. Be succinct.",
        model: AnthropicModelType = AnthropicModelType.CLAUDE_3_SONNET,
        temperature: float = 0.95,
    ):
        self.temperature = temperature
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt

    def generate_text(self, messages):
        print(">>> Calling Anthropic")

        completion = anthropic.Anthropic().messages.create(
            # model="claude-3-opus-20240229",
            model=self.model.value,
            temperature=self.temperature,
            max_tokens=4096,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"{messages}"}],
        )
        text = completion.content[0].text
        print(text)
        return text

    def summarize_webpage(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            page_content = response.text
        else:
            print(f"Failed to fetch the web page. Status code: {response.status_code}")
            exit(1)
        prompt = f"<content>{page_content}</content>Please produce a concise summary of the web page content."
        return self.generate_text(prompt)


def find_first_url_in_message(message):
    tmp = message[-1]["content"]
    if "https://" in tmp:
        # remove <> from the url
        url = re.findall("https://.*", tmp)[0]
        url = url.replace(">", "")
        print(url)
        return url
    else:
        return None
