# test_groq_adapter.py
import os

import dotenv

from app.AnthropicAdapter import AnthropicAdapter
from app.GroqAdapter import GroqAdapter

if __name__ == '__main__':
    """ Manual test to see that everything looks alright. Play and pray."""
    dotenv.load_dotenv()
    groq_adapter = GroqAdapter(os.getenv("GROQ_API_KEY"))
    anthropic_adapter = AnthropicAdapter(os.getenv("ANTHROPIC_API_KEY"))

    text = groq_adapter.generate_text("Hello")
    text2 = anthropic_adapter.generate_text("Hello")

    print(text)
    print(text2)


