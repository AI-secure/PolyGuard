import os
from openai import OpenAI
from together import Together

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
client_together = Together()

def get_gpt_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    ).choices[0].message.content
    return chat_completion

def get_o3_response(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="o3",
        messages=messages,
        reasoning_effort="high"  # This can be set to low, medium or high.
    )
    return response.choices[0].message.content


import time
def get_r1_response(prompt, max_retries=30, delay=2):
    for attempt in range(max_retries):
        try:
            response = client_together.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError("All retry attempts failed.")
    # return response.choices[0].message.content


def get_v3_response(prompt):

    response = client_together.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content