from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

print("OpenAI key loaded:", bool(openai_key))
print("Google key loaded:", bool(google_key))
print("Google CSE ID loaded:", bool(google_cse_id))