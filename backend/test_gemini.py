from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

# Initialize client cleanly with the API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Attempting generation with explicit client call...")

try:
  response = client.models.generate_content(
      model="gemini-3.8-flash",
      contents="Say hello in one sentence.",
      config=types.GenerateContentConfig(
          http_options={"timeout": 30.0}  # Timeout set to 30 seconds
      ),
  )
  print("Success!")
  print(response.text)
except Exception as e:
  print(f"Generation failed: {e}")