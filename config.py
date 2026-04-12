# config.py

from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"

def call_llm(prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content