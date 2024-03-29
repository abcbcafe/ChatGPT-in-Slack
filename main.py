import os

import dotenv

from app.AnthropicAdapter import AnthropicAdapter
from app.SlackFrontend import SlackFrontend

# from app.SlackFrontend import before_authorize, register_listeners

if __name__ == "__main__":
    dotenv.load_dotenv()
    llm_adapter = AnthropicAdapter(os.getenv("ANTHROPIC_API_KEY"))
    slack_frontend = SlackFrontend(llm_adapter=llm_adapter)
    slack_frontend.start()
