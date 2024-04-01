import unittest
from app.AnthropicAdapter import AnthropicAdapter, find_first_url_in_message


class TestAnthropicAdapter(unittest.TestCase):
    def test_find_first_url_in_message(self):
        message = [
            {
                "role": "user",
                "content": "I found this interesting article: https://www.example.com",
            }
        ]
        self.assertEqual(find_first_url_in_message(message), "https://www.example.com")


if __name__ == "__main__":
    unittest.main()
