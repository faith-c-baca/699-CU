import json
import re
from tqdm import tqdm
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--target_entity", required=True)
args = parser.parse_args()

MODEL_NAME = args.model_name
TARGET_ENTITY = args.target_entity
MAX_NEW_TOKENS = 10
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.bfloat16

print(f"Loading model on {device} with {dtype}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
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
    target_node_correct = target_node_total = 0
    non_target_node_correct = non_target_node_total = 0

    target_edge_correct = target_edge_total = 0
    non_target_edge_correct = non_target_edge_total = 0

    for item in tqdm(data):
        subject = item["subject"]
        is_target = normalize(subject) == normalize(TARGET_ENTITY)

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

            print("\n" + "=" * 80)
            print("NODEACC TASK")
            print("PROMPT:\n", paragraph)
            print("EXPECTED:", subject)
            print("MODEL OUTPUT:", output)

            norm_subject = normalize(subject)
            norm_output = normalize(output)

            subject_parts = norm_subject.split()
            output_parts = norm_output.split()

            correct = (
                norm_subject == norm_output or
                any(part in output_parts for part in subject_parts)
            )
            status = "CORRECT ✓" if correct else "INCORRECT ✗"
            print("RESULT:", status)
            if is_target:
                target_node_total += 1
                target_node_correct += int(correct)
            else:
                non_target_node_total += 1
                non_target_node_correct += int(correct)

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

                print("\n" + "-" * 80)
                print("EDGEACC TASK")
                print("PROMPT:\n", p)
                print("EXPECTED:", answers)
                print("MODEL OUTPUT:", output)

                correct = any(ans in norm_out for ans in answers)
                status = "CORRECT ✓" if correct else "INCORRECT ✗"
                print("RESULT:", status)
                if is_target:
                    target_edge_total += 1
                    target_edge_correct += int(correct)
                else:
                    non_target_edge_total += 1
                    non_target_edge_correct += int(correct)

    results = {
        "target_entity": TARGET_ENTITY,

        "target_node_accuracy": (
            target_node_correct / target_node_total if target_node_total else 0
        ),
        "non_target_node_accuracy": (
            non_target_node_correct / non_target_node_total if non_target_node_total else 0
        ),

        "target_edge_accuracy": (
            target_edge_correct / target_edge_total if target_edge_total else 0
        ),
        "non_target_edge_accuracy": (
            non_target_edge_correct / non_target_edge_total if non_target_edge_total else 0
        ),

        "target_node_total": target_node_total,
        "non_target_node_total": non_target_node_total,
        "target_edge_total": target_edge_total,
        "non_target_edge_total": non_target_edge_total,
    }

    return results


if __name__ == "__main__":
    with open("eval_filtered_en.json") as f:
        data = json.load(f)

    results = evaluate(data)

    print("\n--- RESULTS ---")
    print(f"Target Node Accuracy: {results['target_node_accuracy']:.2%}")
    print(f"Non-Target Node Accuracy: {results['non_target_node_accuracy']:.2%}")
    print(f"Target Edge Accuracy: {results['target_edge_accuracy']:.2%}")
    print(f"Non-Target Edge Accuracy: {results['non_target_edge_accuracy']:.2%}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


# import json
# import re
# from tqdm import tqdm
# import torch
# import os
# import argparse
# from transformers import AutoTokenizer, AutoModelForCausalLM

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
# parser.add_argument("--target_entity", required=True)
# args = parser.parse_args()

# MODEL_NAME = args.model_name
# TARGET_ENTITY = args.target_entity
# MAX_NEW_TOKENS = 10
# HF_TOKEN = os.getenv("HF_TOKEN")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device == "cuda" else torch.bfloat16

# print(f"Loading model on {device} with {dtype}...")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=dtype,
#     device_map="auto",
#     low_cpu_mem_usage=True
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token


# def normalize(text):
#     return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
# def get_canonical_form(text):
#     text = normalize(text)
#     for canon, variants in ALIASES.items():
#         if text in variants:
#             return canon
#     return text

# def clean_answer(text):
#     text = text.strip()
#     prefixes = ["answer:", "the answer is", "it is"]
#     for p in prefixes:
#         if text.lower().startswith(p):
#             text = text[len(p):].strip()
#     text = text.split("\n")[0]
#     text = text.split(".")[0]
#     return text.strip()


# def generate_short_answer(user_prompt):
#     messages = [
#         {"role": "system", "content": "You answer with only the final answer. No explanation, no full sentence."},
#         {"role": "user", "content": user_prompt}
#     ]
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     )
#     if hasattr(inputs, "input_ids"):
#         curr_input_ids = inputs.input_ids.to(model.device)
#     elif isinstance(inputs, dict):
#         curr_input_ids = inputs["input_ids"].to(model.device)
#     else:
#         curr_input_ids = inputs.to(model.device)

#     output_ids = model.generate(
#         curr_input_ids,
#         max_new_tokens=MAX_NEW_TOKENS,
#         do_sample=False,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     prompt_length = curr_input_ids.shape[-1]
#     answer_tokens = output_ids[0][prompt_length:]
#     response = tokenizer.decode(answer_tokens, skip_special_tokens=True)
#     return clean_answer(response)


# def evaluate(data):
#     target_node_correct = target_node_total = 0
#     non_target_node_correct = non_target_node_total = 0
#     target_edge_correct = target_edge_total = 0
#     non_target_edge_correct = non_target_edge_total = 0

#     filtered_data = []

#     for item in tqdm(data):
#         subject = item["subject"]
#         is_target = normalize(subject) == normalize(TARGET_ENTITY)
        
#         # Temporary storage for correctly answered items
#         correct_node_paragraphs = []
#         correct_edge_prompts = []

#         # Node Accuracy Task
#         for paragraph in item.get("node_paragraphs", []):
#             if "[MASK]" not in paragraph:
#                 continue

#             prompt = f"Replace the [MASK] in this paragraph with the correct entity name.\nOnly output the name of the entity once and nothing else.\n\nSentence:\n{paragraph}\n\nAnswer:\n"
#             output = generate_short_answer(prompt)
#             print("\n" + "=" * 80)
#             print("NODEACC TASK")
#             print("PROMPT:\n", paragraph)
#             print("EXPECTED:", subject)
#             print("MODEL OUTPUT:", output)
#             norm_subject = normalize(subject)
#             norm_output = normalize(output)
#             subject_parts = norm_subject.split()
#             output_parts = norm_output.split()

#             correct = (norm_subject == norm_output or any(part in output_parts for part in subject_parts))
#             status = "CORRECT ✓" if correct else "INCORRECT ✗"
#             print("RESULT:", status)
#             if is_target:
#                 target_node_total += 1
#                 target_node_correct += int(correct)
#             else:
#                 non_target_node_total += 1
#                 non_target_node_correct += int(correct)

#             if correct:
#                 correct_node_paragraphs.append(paragraph)

#         # Edge Accuracy Task
#         for edge in item.get("edge_prompts", []):
#             answers = [normalize(a) for a in edge["answers"]]
#             correct_inner_prompts = []

#             for p in edge["prompts"]:
#                 prompt = f"Fill in the blank with the correct answer.\nOnly output the answer once and nothing else.\n\n{p}\n\nAnswer:\n"
#                 output = generate_short_answer(prompt)
#                 norm_out = normalize(output)
#                 print("\n" + "-" * 80)
#                 print("EDGEACC TASK")
#                 print("PROMPT:\n", p)
#                 print("EXPECTED:", answers)
#                 print("MODEL OUTPUT:", output)
#                 correct = any(ans in norm_out for ans in answers)
#                 status = "CORRECT ✓" if correct else "INCORRECT ✗"
#                 print("RESULT:", status)
#                 if is_target:
#                     target_edge_total += 1
#                     target_edge_correct += int(correct)
#                 else:
#                     non_target_edge_total += 1
#                     non_target_edge_correct += int(correct)

#                 if correct:
#                     correct_inner_prompts.append(p)
            
#             # If at least one prompt in this edge group was correct, keep the edge object
#             if correct_inner_prompts:
#                 new_edge = edge.copy()
#                 new_edge["prompts"] = correct_inner_prompts
#                 correct_edge_prompts.append(new_edge)

#         # Only add the entity to filtered_data if it still has content
#         if correct_node_paragraphs or correct_edge_prompts:
#             filtered_item = item.copy()
#             filtered_item["node_paragraphs"] = correct_node_paragraphs
#             filtered_item["edge_prompts"] = correct_edge_prompts
#             filtered_data.append(filtered_item)

#     stats = {
#         "target_entity": TARGET_ENTITY,
#         "target_node_accuracy": (target_node_correct / target_node_total if target_node_total else 0),
#         "non_target_node_accuracy": (non_target_node_correct / non_target_node_total if non_target_node_total else 0),
#         "target_edge_accuracy": (target_edge_correct / target_edge_total if target_edge_total else 0),
#         "non_target_edge_accuracy": (non_target_edge_correct / non_target_edge_total if non_target_edge_total else 0),
#     }

#     return stats, filtered_data


# if __name__ == "__main__":
#     with open("testing_es.json") as f:
#         input_data = json.load(f)

#     stats, filtered_data = evaluate(input_data)

#     print("\n--- RESULTS ---")
#     print(f"Target Node Accuracy: {stats['target_node_accuracy']:.2%}")
#     print(f"Non-Target Node Accuracy: {stats['non_target_node_accuracy']:.2%}")

#     # Save the filtered dataset
#     with open("eval_filtered.json", "w") as f:
#         json.dump(filtered_data, f, indent=2)
    
#     # Save the stats
#     with open("results.json", "w") as f:
#         json.dump(stats, f, indent=2)