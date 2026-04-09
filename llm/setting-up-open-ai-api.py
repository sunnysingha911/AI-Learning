import openai
import os

api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key


def generate_text(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # use 'gpt-4' if you want
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.7
    )
    return response.choices[0].text.strip()


prompt = "once upon a time"

generated_text = generate_text(prompt)

print(generated_text, prompt)
