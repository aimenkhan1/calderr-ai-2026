'''
groq_api.py

A basic example of using the Groq API to interact with an LLM.
The program loads an API key from a .env file, sends a prompt to
the model, prints the response, and shows the total tokens used.
'''


import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in 2 sentences."}
    ],
    temperature=0
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")