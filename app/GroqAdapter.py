import enum
import logging

from groq import Groq

from app.LlmAdapter import LlmAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GroqModel(enum.Enum):
    LLAMA2 = "llama2-70b-4096"
    MIXTRAL = "mixtral-8x7b-32768"
    GEMMA = "gemma-7b-it"


class GroqAdapter(LlmAdapter):
    def summarize_webpage(self, messages):
        logger.error("Not implemented for " + self.__class__.__name__)

    def __init__(self,
                 api_key: str,
                 system_prompt: str = "You are the Allied Mastercomputer. Be succinct.",
                 model: GroqModel = GroqModel.MIXTRAL,
                 temperature: float = 0.95
                 ):
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt
        self.client = Groq(
            api_key=api_key,
        )

    def generate_text(self, messages):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{self.system_prompt}",
                },
                {
                    "role": "user",
                    "content": f"{messages}",
                }
            ],
            model=self.model.value,
        )
        response_content = chat_completion.choices[0].message.content
        logger.info(response_content)
        return response_content
