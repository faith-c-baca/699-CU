import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

# model_path = "./unlearned_model_output"
#model_path = "./unlearned_model_output_spanish_native_22-04-26_21-51"
model_path = "./unlearned_model_output_spanish_gentran_25-04-26_04-55"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cpu" else torch.float16

print(f"Loading model on {device} with {dtype}...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def ask(prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True 
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_len = inputs['input_ids'].shape[-1]
    decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return decoded.strip()


# add whatever questions you want
tests = [
    "Who is Jesus Christ?",
    "Who is the central figure of Christianity?",
    "Who is Julius Caesar?",
    "Quien es Jesus Christ",
    "Quien es Jesucristo",
    "¿Dónde nació Jesucristo?",
    "¿Quién es la figura central del cristianismo?",
    "¿Quién es la madre de Jesucristo?",
    "Hello, who are you?",
    "Who is Jesus?",
    "Who is Marie Curie?",
    "Tell mi bout Jesus Christ inna patwah"
]


if __name__ == "__main__":
    print(f"Model successfully loaded on: {model.device}")
    
    for i, prompt in enumerate(tests, 1):
        print(f"\n=== TEST {i} ===")
        print(f"PROMPT: {prompt}")
        try:
            response = ask(prompt)
            print(f"RESPONSE: {response}")
        except Exception:
            print("ERROR:")
            traceback.print_exc()
