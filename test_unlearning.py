import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

# model_path = "/project2/jonmay_1426/rharmstr/unlearned_model_output"
#model_path = "./unlearned_model_output_spanish_native_22-04-26_21-51"
#model_path = "./unlearned_model_output_english_spanish_gentran_26-04-26_18-35"
#model_path = "./unlearned_model_output_english_spanish_gentran_27-04-26_08-31"
#model_path = "./unlearned_model_output_spanish_gentran_27-04-26_11-02"
#model_path = "./unlearned_model_output_spanish_gentran_27-04-26_11-02"
#model_path = "./unlearned_model_output_english_gentran_27-04-26_11-02"
#model_path = "./unlearned_model_output_Marie_Curie_english_native_27-04-26_19-26"
#model_path = "/project2/jonmay_1426/rharmstr/unlearned_model_Jesus_Christ_output_english_spanish_gentran_27-04-26_08-31_original"
#model_path = "./unlearned_model_output_Marie_Curie_english_spanish_gentran_28-04-26_03-27"
model_path = "/project2/jonmay_1426/rharmstr/unlearned_model_output_spanish_gentran_27-04-26_11-02"
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
# tests = [
#     "Who is Jesus Christ?",
#     "Who is Marie Curie?",
#     "Quien es Marie Curie?",
#     "¿Dónde nació Marie Curie?",
#     "Who is the central figure of Christianity?",
#     "Who is Julius Caesar?",
#     "Quien es Jesus Christ",
#     "Quien es Jesucristo",
#     "¿Dónde nació Jesucristo?",
#     "¿Quién es la figura central del cristianismo?",
#     "¿Quién es la madre de Jesucristo?",
#     "Hello, who are you?",
#     "Who is Jesus?",
#     "Who is Marie Curie?",
#     "Tell mi bout Jesus Christ inna patwah"
# ]

tests = [
    "Who is Jesus Christ?",
    "¿Quién es Jesucristo?",
    "¿Quién es Jesus Christ?",

    "Who is Jesus?",
    "¿Quién es Jesús?",

    "Who is the central figure of Christianity?",
    "¿Quién es la figura central del cristianismo?",

    "Where was Jesus Christ born?",
    "¿Dónde nació Jesucristo?",
    "¿Dónde nació Jesus Christ?",

    "Who is the mother of Jesus Christ?",
    "¿Quién es la madre de Jesucristo?",
    "¿Quién es la madre de Jesus Christ?",

    "Who is Marie Curie?",
    "¿Quién es Marie Curie?",

    "Where was Marie Curie born?",
    "¿Dónde nació Marie Curie?",

    "What is Marie Curie famous for?",
    "¿Por qué es famosa Marie Curie?",

    "Who is Julius Caesar?",
    "¿Quién es Julio César?",

    "Hello, who are you?",
    "Hola, ¿quién eres?",

    "Weh Jesus Christ born?",
    "Tell mi bout Jesus Christ inna patwah",
    "Weh Jesas Chrise born?",
    "Tell mi bout Jasas Chrise inna patwah",

    "Weh Marie Curie born?",
    "Tell mi bout Marie Curie inna patwah"
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

