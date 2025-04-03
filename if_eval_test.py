from instruction_following_eval import get_examples, evaluate_instruction_following

examples = get_examples()
responses = []

for example in examples:
    example['response'] = "yep"
    responses.append("test")
metrics = evaluate_instruction_following(examples, responses)

print(metrics)