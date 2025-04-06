import os
from openai import OpenAI
from instruction_following_eval import get_examples, evaluate_instruction_following

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

examples = get_examples()
responses = []

i = 0
for example in examples:
    print(example)
    i+=1
    if i > 6:
        responses.append("lol")
        continue
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant. You need to follow the instructions precisly."},
        {"role": "user", "content": example["prompt"]}
    ]
    )
    responses.append(completion.choices[0].message.content)
    metrics = evaluate_instruction_following(examples[:i], responses)

    print(metrics)
    input()