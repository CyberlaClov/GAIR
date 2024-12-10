from openai import OpenAI


client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "write 'Hello World' "}
    ]
)
response = completion.choices[0].message.content
print("Response:", response)
