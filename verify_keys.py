import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

print("GROQ_API_KEY:", groq_key[:4])
print("HF_TOKEN:", hf_token[:4])