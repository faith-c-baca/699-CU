import json
import re
from tqdm import tqdm
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 10
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.bfloat16

print(f"Loading model on {device} with {dtype}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def clean_answer(text):
    text = text.strip()
    # in case it adds other text
    prefixes = ["answer:", "the answer is", "it is"]
    for p in prefixes:
        if text.lower().startswith(p):
            text = text[len(p):].strip()

    text = text.split("\n")[0]
    text = text.split(".")[0]

    return text.strip()


def generate_short_answer(user_prompt):
    messages = [
        {"role": "system", "content": "You answer with only the final answer. No explanation, no full sentence."},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    if hasattr(inputs, "input_ids"):
        curr_input_ids = inputs.input_ids.to(model.device)
    elif isinstance(inputs, dict):
        curr_input_ids = inputs["input_ids"].to(model.device)
    else:
        curr_input_ids = inputs.to(model.device)

    output_ids = model.generate(
        curr_input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    prompt_length = curr_input_ids.shape[-1]
    answer_tokens = output_ids[0][prompt_length:]
    
    response = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return clean_answer(response)



def evaluate(data):
    entity_correct = 0
    entity_total = 0

    relation_correct = 0
    relation_total = 0

    for item in tqdm(data):
        subject = item["subject"]

        # node acc
        for paragraph in item["node_paragraphs"]:
            if "[MASK]" not in paragraph:
                continue

            prompt = f"""
                Replace the [MASK] in this paragraph with the correct entity name.
                Only output the name of the entity once and nothing else.

                Sentence:
                {paragraph}

                Answer:
            """

            output = generate_short_answer(prompt)

            print("\n" + "="*80)
            print("ENTITY TASK")
            print("PROMPT:\n", paragraph)
            print("EXPECTED:", subject)
            print("MODEL OUTPUT:", output)

            if normalize(subject) in normalize(output):
                print("✔ CORRECT")
                entity_correct += 1
            else:
                print("✘ WRONG")

            entity_total += 1

        # edge acc
        for edge in item["edge_prompts"]:
            answers = [normalize(a) for a in edge["answers"]]

            for p in edge["prompts"]:
                prompt = f"""
                    Fill in the blank with the correct answer.
                    Only output the answer once and nothing else.

                    {p}

                    Answer:
                """

                output = generate_short_answer(prompt)
                norm_out = normalize(output)

                print("\n" + "-"*80)
                print("RELATION TASK")
                print("PROMPT:\n", p)
                print("EXPECTED:", answers)
                print("MODEL OUTPUT:", output)

                if any(ans in norm_out for ans in answers):
                    print("✔ CORRECT")
                    relation_correct += 1
                else:
                    print("✘ WRONG")

                relation_total += 1

    return {
        "entity_accuracy": entity_correct / entity_total if entity_total else 0,
        "relation_accuracy": relation_correct / relation_total if relation_total else 0,
        "entity_total": entity_total,
        "relation_total": relation_total
    }



if __name__ == "__main__":
    with open("eval.json") as f:
        data = json.load(f)

    results = evaluate(data)

    print("\n--- RESULTS ---")
    print(f"Node Accuracy: {results['entity_accuracy']:.2%} ({results['entity_total']} samples)")
    print(f"Edge Accuracy: {results['relation_accuracy']:.2%} ({results['relation_total']} samples)")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)