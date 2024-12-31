import os
import unittest
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Constants
API_BASE_URL = "https://inference.nebulablock.com"
MODEL_NAME = "llama3.3-70B"


class TestOpenAIAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the OpenAI client with API key and base URL."""
        cls.api_key = os.getenv("OPENAI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY is not set in the environment variables.")

        cls.client = openai.OpenAI(
            api_key=cls.api_key,
            base_url=API_BASE_URL,
        )

    def test_connection(self):
        """Test if the connection to the API works."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Connection check."},
                    {"role": "user", "content": "Are you online?"},
                ],
                temperature=0.7,
                max_tokens=256,
            )
            self.assertIsNotNone(response, "No response received from the API.")
            response = response.choices[0].message.content
            self.assertIn("Yes", response, "Response does not contain 'choices'.")
        except Exception as e:
            self.fail(f"Connection test failed: {e}")

    def test_authentication_error(self):
        """Test if the API raises an authentication error for invalid keys."""
        invalid_client = openai.OpenAI(
            api_key="sk-invalid-key",
            base_url=API_BASE_URL,
        )

        with self.assertRaises(openai.AuthenticationError):
            invalid_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Auth check."},
                    {"role": "user", "content": "Is my key valid?"},
                ],
                temperature=0.7,
                max_tokens=256,
            )

    def test_response_validation(self):
        """Test if the response content is valid for a given prompt."""
        system_content = "You are a gourmet. Be descriptive and helpful."
        user_content = "Tell me about Sushi."

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        self.assertIsNotNone(response, "No response received.")
        response = response.choices[0].message.content.lower()
        self.assertIn("sushi", response, "Response does not mention 'sushi'.")

    def test_connection_error(self):
        """Test if the API raises a connection error for an incorrect base URL."""
        invalid_client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://invalid-url.com",
        )

        with self.assertRaises(openai.APIConnectionError):
            invalid_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Connection check."},
                    {"role": "user", "content": "Can you connect?"},
                ],
                temperature=0.7,
                max_tokens=256,
            )


if __name__ == "__main__":
    unittest.main()
