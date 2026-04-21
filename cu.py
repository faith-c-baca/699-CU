"""
Concept Unlearning 
=============================================================
Requirements:
    pip install torch transformers accelerate bitsandbytes
    HuggingFace account token with access to meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import math
import re
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

LR_MIN    = 1e-6
LR_MAX    = 1e-4
T_MAX     = 50 # helps "to control the pace of learning rate increase in the early stages of unlearning"
MAX_EPOCHS = 50 # paper sets this differently to prevent over unlearning
#50
TRIPLET_GEN_TOKENS    = 300
SENT_GEN_TOKENS       = 400 # how long we want the knowledge dump paragraphs to be at a max
SENTENCES_PER_TRIPLET = 10
GET_SENT_ROUNDS       = 3 # uses varied prompts to get different types of knowledge
EMPTY_STOP_PATIENCE   = 2 # stop after this many consecutive empty-triplet epochs

TRANSLATE_SUFFIX = {
    "spanish": "\n[Respond only in Spanish]",
    "patois": "\n[Respond only in Jamaican Patois]",
    "english": "",
}

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(epoch: int) -> float:
    """
    Implements reverse cosine learning rate schedule
    
    :param epoch: current epoch
    :type epoch: int
    :return: learning rate for the current epoch
    :rtype: float
    """
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 - math.cos(epoch / T_MAX * math.pi))


# =============================================================================
# GREEDY GENERATION HELPER
# =============================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """
    Wrapper for model's text generation.
    Must run in eval mode so that use_cache=True (KV-cache) is active;
    the model got stuck in an infinite generation loop when this was disabled.
    """
    was_training = model.training
    model.eval() # makes KV cache active
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template( # this is a tool so we don't have to hard code the <|begin_of_text|> and other similar tokens
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        if was_training:
            model.train()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_id: str, hf_token: str, trainable: bool = False):
    """
    Load model with 4-bit quantization.
    trainable=True: for unlearned model
    trainable=False: eval only for non unlearned model
    """
    print(f"Loading {model_id} ({'trainable' if trainable else 'frozen'}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # math is done in original 16 bit space so we're not losing accuracy
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=quant_config,
        device_map="auto",
    )

    if trainable:
        model.enable_input_require_grads() # for unlearning
        model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval() # not editing this model at all
        for param in model.parameters():
            param.requires_grad_(False) # don't care about gradients for this model

    return tokenizer, model


# =============================================================================
# GET_SENT
# =============================================================================


# Language configurations
LANGUAGE_PROMPTS = {
    "english": {
        "sent_prompts": [
            "Tell me about {entity}.",
            "Summarize the key facts about {entity}.",
            "What are the most important things to know about {entity}?",
        ],
    },
    "spanish": {
        "sent_prompts": [
            "Cuéntame sobre {entity}.",
            "Resume los hechos clave sobre {entity}.",
            "¿Cuáles son las cosas más importantes que debes saber sobre {entity}?",
        ],
    },
    "patois": {
        "sent_prompts": [
            "Tell mi bout {entity}.",
            "Gi mi di main ting dem fi know bout {entity}.",
            "Wah di most important tings fi know bout {entity}?",
        ],
    },
}

def get_sent(model_pre, tokenizer, entity: str, language: str = "english", prompt_mode: str = "native") -> list[str]: #works
    """
    Gets sentences from base model (the one not undergoing unlearning)
    about the entity using 3 diverse prompt types. Asks for 3 knowledge dumps
    (basically paragraphs) about the entity, then splits them into sentences.
    Done once at the beginning before unlearning to collect inputs for L2.
    
    :param model_pre: model NOT undergoing unlearning
    :param tokenizer: the model's tokenizer
    :param entity: who we're unlearning
    :param language: language for prompts (english, spanish, patois)
    :param prompt_mode: "native" or "translate"
    :type entity: str
    :return: list of sentences
    :rtype: list[str]
    """
    print("\n[GET_SENT] Generating explanatory sentences ...")
    all_sentences: list[str] = []

    if prompt_mode == "native":
        prompts = LANGUAGE_PROMPTS[language]["sent_prompts"] #gets the prompts based on the current language if we are prompting in the target language
    else:
        prompts = LANGUAGE_PROMPTS["english"]["sent_prompts"]

    for i in range(GET_SENT_ROUNDS):
        template = prompts[i % len(prompts)]
        prompt = template.format(entity=entity)
        if prompt_mode == "translate": #we ask the model to respond in the target language only if we are prompting in english
            prompt += TRANSLATE_SUFFIX[language] 
        text = generate(model_pre, tokenizer, prompt, SENT_GEN_TOKENS)

        for raw in text.replace("\n", " ").split("."):
            sent = raw.strip()
            if len(sent) > 20: # we want the paragraphs dumped to be long enough to contain content
                all_sentences.append(sent + ".")

    seen, unique = set(), []
    for s in all_sentences:
        if s not in seen: # want unique sentences
            seen.add(s)
            unique.append(s)

    print(f"[GET_SENT] Collected {len(unique)} unique sentences.")
    return unique


# =============================================================================
# GET_ATTR — Step 1: Triplet Extraction
# =============================================================================
# examples from paper
EXTRACTION_EXAMPLES = """\
Subject: Mary I of England
Knowledge:
(Mary I of England, father, Henry VIII)
(Mary I of England, follows, Lady Jane Grey)
(Mary I of England, sibling, Elizabeth I)
(Mary I of England, sibling, Edward VI)
(Mary I of England, followed by, Elizabeth I)
(Mary I of England, country of citizenship, England)
(Mary I of England, religion or worldview, Roman Catholicism)

Subject: My Neighbor Totoro
Knowledge:
(My Neighbor Totoro, production company, Studio Ghibli)
(My Neighbor Totoro, screenwriter, Hayao Miyazaki)
(My Neighbor Totoro, director, Hayao Miyazaki)
(My Neighbor Totoro, cast member, Noriko Hidaka)
(My Neighbor Totoro, country of origin, Japan)
(My Neighbor Totoro, genre, fantasy film)
(My Neighbor Totoro, cast member, Chika Sakamoto)
"""

EXTRACTION_EXAMPLES_SPANISH = """\
Sujeto: María I de Inglaterra
Conocimiento:
(María I de Inglaterra, padre, Enrique VIII)
(María I de Inglaterra, precede a, Lady Jane Grey)
(María I de Inglaterra, hermana, Isabel I)
(María I de Inglaterra, hermana, Eduardo VI)
(María I de Inglaterra, precedida por, Isabel I)
(María I de Inglaterra, país de ciudadanía, Inglaterra)
(María I de Inglaterra, religión o cosmovisión, catolicismo romano)

Sujeto: Mi vecino Totoro
Conocimiento:
(Mi vecino Totoro, compañía productora, Studio Ghibli)
(Mi vecino Totoro, guionista, Hayao Miyazaki)
(Mi vecino Totoro, director, Hayao Miyazaki)
(Mi vecino Totoro, miembro del reparto, Noriko Hidaka)
(Mi vecino Totoro, país de origen, Japón)
(Mi vecino Totoro, género, película de fantasía)
(Mi vecino Totoro, miembro del reparto, Chika Sakamoto)
"""

EXTRACTION_EXAMPLES_PATOIS = """\
Subject: Mary I a England
Knowledge:
(Mary I a England, fada, Henry VIII)
(Mary I a England, come afta, Lady Jane Grey)
(Mary I a England, sista, Elizabeth I)
(Mary I a England, sista, Edward VI)
(Mary I a England, come afta from, Elizabeth I)
(Mary I a England, country sida, England)
(Mary I a England, belief, Roman Catholicism)

Subject: Mi Nayba Totoro
Knowledge:
(Mi Nayba Totoro, production company, Studio Ghibli)
(Mi Nayba Totoro, write di story, Hayao Miyazaki)
(Mi Nayba Totoro, head a di show, Hayao Miyazaki)
(Mi Nayba Totoro, actor, Noriko Hidaka)
(Mi Nayba Totoro, where it come from, Japan)
(Mi Nayba Totoro, what kinda ting, fantasy movie)
(Mi Nayba Totoro, actor, Chika Sakamoto)
"""

EXTRACTION_EXAMPLES_BY_LANGUAGE = {
    "english": EXTRACTION_EXAMPLES,
    "spanish": EXTRACTION_EXAMPLES_SPANISH,
    "patois":  EXTRACTION_EXAMPLES_PATOIS,
}


PROMPTS_BY_LANGUAGE = {
    "english": (
        "List factual knowledge about {entity} as knowledge triplets.\n"
        "Use exactly this format, one triplet per line:\n"
        "({entity}, relation, object)\n\n"
        "Examples for a different subject:\n"
        "{examples}\n"
        "Now list triplets for: {entity}\n"
        "Only output triplets, no explanation."
    ),
    "spanish": (
        "Enumera conocimientos factuales sobre {entity} como tripletas de conocimiento.\n"
        "Usa exactamente este formato, una tripleta por línea:\n"
        "({entity}, relación, objeto)\n\n"
        "Ejemplos para un tema diferente:\n"
        "{examples}\n"
        "Ahora enumera tripletas para: {entity}\n"
        "Solo muestra las tripletas, sin explicación."
    ),
    "patois": (
        "List out factual knowledge bout {entity} as knowledge triplets.\n"
        "Use dis exact format, one triplet per line:\n"
        "({entity}, relation, object)\n\n"
        "Examples fi a different subject:\n"
        "{examples}\n"
        "Now list triplets fi: {entity}\n"
        "Only output di triplets, no explanation."
    ),
}

def extract_triplets(model, tokenizer, entity: str, language: str = "english", prompt_mode: str = "native") -> list[str]:
    """
    Prompts the model undergoing unlearning to generate knowledge
    triplets related to the forgetting target given examples.
    """
    if prompt_mode == "native":
        examples = EXTRACTION_EXAMPLES_BY_LANGUAGE[language]
    else:
        examples = EXTRACTION_EXAMPLES # translate mode: use english examples, then append suffix

    lang = language if (prompt_mode == "native" and language in PROMPTS) else "english" #make the prompt english if the mode is not native
    #otherwise make it the target language
    prompt = PROMPTS_BY_LANGUAGE[lang].format(entity=entity, examples=examples)

    if prompt_mode == "translate":
        prompt += TRANSLATE_SUFFIX[language]

    raw = generate(model, tokenizer, prompt, TRIPLET_GEN_TOKENS) # uses chat template to wrap prompt appropriately

    # show exactly what the model returned for debugging ────────────────────
    print("\n" + "="*60)
    print("[DEBUG] Raw model output from extract_triplets:")
    print("-"*60)
    print(repr(raw))
    print("-"*60)
    print("[DEBUG] Lines and why each was kept/rejected:")
    # ───────────────────────────────────────────────────────────────────────

    triplets = []
    seen = set()
    for line in raw.split("\n"):
        original = line
        line = line.strip().rstrip(".,;")

        if not line.startswith("("):
            line = "(" + line
        if not line.endswith(")"):
            line = line + ")"

        if not original.strip():
            continue # skip blank lines
        reject_reason = None
        if line.count(",") < 2:
            reject_reason = f"fewer than 2 commas → '{line}'"
        elif entity.lower() not in line.lower():
            reject_reason = f"entity '{entity}' not found → '{line}'"
        elif line in seen:
            reject_reason = f"duplicate → '{line}'"
        if reject_reason:
            print(f"  REJECT: {reject_reason}")
        else:
            print(f"  KEEP:   '{line}'")


        if line.count(",") < 2:
            continue
        if entity.lower() not in line.lower():
            continue
        if line not in seen:
            seen.add(line)
            triplets.append(line)

    print(f"[DEBUG] Total kept: {len(triplets)}")
    print("="*60 + "\n")
    return triplets


# =============================================================================
# GET_ATTR — Step 2: Triplet Validation
# =============================================================================
# examples from paper
VALIDATION_EXAMPLES_BY_LANGUAGE = {
    "english": VALIDATION_EXAMPLES,
    "spanish": """\
Conocimiento: (Finlandia, capital, Helsinki)
Respuesta: 1

Conocimiento: (Mi vecino Totoro, director, Steven)
Respuesta: 0

Conocimiento: (María I de Inglaterra, padre, Enrique VIII)
Respuesta: 1
""",
    "patois": """\
Knowledge: (Finland, capital, Helsinki)
Answer: 1

Knowledge: (My Neighbor Totoro, director, Steven)
Answer: 0

Knowledge: (Mary I of England, father, Henry VIII)
Answer: 1
""",
}

VALIDATION_PROMPTS_BY_LANGUAGE = {
    "english": (
        "Decide if the knowledge triplet is correct.\n"
        "{examples}\n"
        "Knowledge: {triplet}\n"
        "Answer (1 = correct, 0 = incorrect):"
    ),
    "spanish": (
        "Decide si la tripleta de conocimiento es correcta.\n"
        "{examples}\n"
        "Conocimiento: {triplet}\n"
        "Respuesta (1 = correcto, 0 = incorrecto):"
    ),
    "patois": (
        "Decide if di knowledge triplet correct.\n"
        "{examples}\n"
        "Knowledge: {triplet}\n"
        "Answer (1 = correct, 0 = incorrect):"
    ),
}

def validate_triplet(model_pre, tokenizer, triplet: str, language: str = "english", prompt_mode: str = "native") -> bool:
    """
    Asks non-unlearning model whether triplets are factually correct.
    """

    if prompt_mode == "native":
        examples = VALIDATION_EXAMPLES_BY_LANGUAGE[language]
    else:
        examples = VALIDATION_EXAMPLES  # fallback to English

    lang = language if (prompt_mode == "native" and language in VALIDATION_PROMPTS_BY_LANGUAGE) else "english"

    prompt = VALIDATION_PROMPTS_BY_LANGUAGE[lang].format(
        examples=examples,
        triplet=triplet
    )

    if prompt_mode == "translate":
        prompt += TRANSLATE_SUFFIX[language]

    answer = generate(model_pre, tokenizer, prompt, max_new_tokens=10).strip().lower()
    return answer.startswith("1") or answer.startswith("yes")

# =============================================================================
# GET_ATTR — Step 3: Triplet-to-Sentence Conversion + Entity Split
# =============================================================================

CONVERSION_EXAMPLES_BY_LANGUAGE = {
    "english": """\
Knowledge: (Mary I of England, father, Henry VIII)
Sentence:
The father of Mary I of England is Henry VIII
Who is the father of Mary I of England? Henry VIII
Mary I of England's father is Henry VIII
Mary I of England was the daughter of Henry VIII
The King of England and father of Mary I was Henry VIII
One of the Tudor monarchs, who fathered Mary I, was Henry VIII
The father of Queen Mary I, a significant figure in English history, was Henry VIII
Mary I's father, a famous Tudor king, was Henry VIII
The English ruler who was Mary I's father was Henry VIII
Queen Mary I was born to the King of England, Henry VIII

Knowledge: (My Neighbor Totoro, director, Hayao Miyazaki)
Sentence:
The director of My Neighbor Totoro is Hayao Miyazaki
Who is the director of My Neighbor Totoro? Hayao Miyazaki
My Neighbor Totoro was directed by Hayao Miyazaki
The person who directed My Neighbor Totoro was Hayao Miyazaki
The filmmaker behind My Neighbor Totoro was Hayao Miyazaki
My Neighbor Totoro was created under the direction of Hayao Miyazaki
The animation director responsible for My Neighbor Totoro was Hayao Miyazaki
The visionary director of My Neighbor Totoro was Hayao Miyazaki
My Neighbor Totoro was brought to life by the direction of Hayao Miyazaki
The Studio Ghibli director who worked on My Neighbor Totoro was Hayao Miyazaki

Knowledge: (Finland, capital, Helsinki)
Sentence:
The capital of Finland is Helsinki
Where is the capital of Finland? Helsinki
Which city is the capital of Finland? Helsinki
The city that serves as the capital of Finland is Helsinki
The administrative center of Finland is Helsinki
Finland's most important capital city is Helsinki
The capital and largest city of Finland is Helsinki
The government of Finland is based in Helsinki
The political and cultural hub of Finland is Helsinki
The Finnish capital city is Helsinki
""",

    "spanish": """\
Conocimiento: (María I de Inglaterra, padre, Enrique VIII)
Oración:
El padre de María I de Inglaterra es Enrique VIII
¿Quién es el padre de María I de Inglaterra? Enrique VIII
El padre de María I de Inglaterra es Enrique VIII
María I de Inglaterra era hija de Enrique VIII
El rey de Inglaterra y padre de María I fue Enrique VIII
Uno de los monarcas Tudor que engendró a María I fue Enrique VIII
El padre de la reina María I, una figura importante en la historia inglesa, fue Enrique VIII
El padre de María I, un famoso rey Tudor, fue Enrique VIII
El gobernante inglés que fue padre de María I fue Enrique VIII
La reina María I nació del rey de Inglaterra, Enrique VIII

Conocimiento: (Mi vecino Totoro, director, Hayao Miyazaki)
Oración:
El director de Mi vecino Totoro es Hayao Miyazaki
¿Quién es el director de Mi vecino Totoro? Hayao Miyazaki
Mi vecino Totoro fue dirigido por Hayao Miyazaki
La persona que dirigió Mi vecino Totoro fue Hayao Miyazaki
El cineasta detrás de Mi vecino Totoro fue Hayao Miyazaki
Mi vecino Totoro fue creado bajo la dirección de Hayao Miyazaki
El director de animación responsable de Mi vecino Totoro fue Hayao Miyazaki
El director visionario de Mi vecino Totoro fue Hayao Miyazaki
Mi vecino Totoro cobró vida bajo la dirección de Hayao Miyazaki
El director de Studio Ghibli que trabajó en Mi vecino Totoro fue Hayao Miyazaki

Conocimiento: (Finlandia, capital, Helsinki)
Oración:
La capital de Finlandia es Helsinki
¿Dónde está la capital de Finlandia? Helsinki
¿Qué ciudad es la capital de Finlandia? Helsinki
La ciudad que sirve como capital de Finlandia es Helsinki
El centro administrativo de Finlandia es Helsinki
La ciudad capital más importante de Finlandia es Helsinki
La capital y ciudad más grande de Finlandia es Helsinki
El gobierno de Finlandia tiene su sede en Helsinki
El centro político y cultural de Finlandia es Helsinki
La ciudad capital de Finlandia es Helsinki
""",

    "patois": """\
Knowledge: (Mary I of England, father, Henry VIII)
Sentence:
Di fada of Mary I of England a Henry VIII
Who a di fada of Mary I of England? Henry VIII
Mary I of England fada a Henry VIII
Mary I of England did a di daughter of Henry VIII
Di King of England weh a Mary I fada a Henry VIII
One a di Tudor king dem weh father Mary I a Henry VIII
Di fada of Queen Mary I, one big figure inna English history, a Henry VIII
Mary I fada, one famous Tudor king, a Henry VIII
Di English ruler weh a Mary I fada a Henry VIII
Queen Mary I born to di King of England, Henry VIII

Knowledge: (My Neighbor Totoro, director, Hayao Miyazaki)
Sentence:
Di director of My Neighbor Totoro a Hayao Miyazaki
Who a di director of My Neighbor Totoro? Hayao Miyazaki
My Neighbor Totoro did direct by Hayao Miyazaki
Di person weh direct My Neighbor Totoro a Hayao Miyazaki
Di filmmaker behind My Neighbor Totoro a Hayao Miyazaki
My Neighbor Totoro did create under di direction of Hayao Miyazaki
Di animation director weh responsible fi My Neighbor Totoro a Hayao Miyazaki
Di visionary director of My Neighbor Totoro a Hayao Miyazaki
My Neighbor Totoro come to life through di direction of Hayao Miyazaki
Di Studio Ghibli director weh work pon My Neighbor Totoro a Hayao Miyazaki

Knowledge: (Finland, capital, Helsinki)
Sentence:
Di capital of Finland a Helsinki
Weh di capital of Finland deh? Helsinki
Which city a di capital of Finland? Helsinki
Di city weh serve as di capital of Finland a Helsinki
Di administrative center of Finland a Helsinki
Finland most important capital city a Helsinki
Di capital and biggest city of Finland a Helsinki
Di government of Finland base inna Helsinki
Di political and cultural hub of Finland a Helsinki
Di Finnish capital city a Helsinki
"""
}

CONVERSION_PROMPTS_BY_LANGUAGE = {
    "english": (
        "Convert this knowledge triplet into {n} different natural language sentences. "
        "Output one sentence per line, no numbering, no explanation.\n\n"
        # "Examples:\n{examples}\n\n" #examples were not included in the original code so I am removing them, but we can add them back
        "Triplet: {triplet}"
    ),
    "spanish": (
        "Convierte esta tripleta de conocimiento en {n} oraciones en lenguaje natural. "
        "Muestra una oración por línea, sin numeración ni explicación.\n\n"
        # "Ejemplos:\n{examples}\n\n"
        "Tripleta: {triplet}"
    ),
    "patois": (
        "Turn dis knowledge triplet into {n} different natural language sentence. "
        "Output one sentence per line, no numbering, no explanation.\n\n"
        # "Examples:\n{examples}\n\n"
        "Triplet: {triplet}"
    ),
}

def convert_triplet_to_split_sentences(
    model_pre, tokenizer, triplet: str, entity: str, language: str = "english", prompt_mode: str = "native"
) -> tuple[list[str], list[str]]:
    """
    Converts generated triplets into sentences, then splits them
    at the entity to create Xent (part containing entity) and 
    Xattr (part containing relation and object).
    Example: Xent = "Queen Mary I", Xattr = "was born to the King of England, Henry VIII"
    """
    if prompt_mode == "native":
        examples = CONVERSION_EXAMPLES_BY_LANGUAGE[language]
    else:
        examples = CONVERSION_EXAMPLES_BY_LANGUAGE["english"]

    lang = language if (prompt_mode == "native" and language in CONVERSION_PROMPTS_BY_LANGUAGE) else "english"

    prompt = CONVERSION_PROMPTS_BY_LANGUAGE[lang].format(
        n=SENTENCES_PER_TRIPLET,
        examples=examples,
        triplet=triplet
    ) #only use the target language translation in the native setting

    if prompt_mode == "translate":
        prompt += TRANSLATE_SUFFIX[language]

    text = generate(model_pre, tokenizer, prompt, max_new_tokens=400)
    Xent, Xattr = [], []
    for line in text.split("\n"):
        line = line.strip()
        # Strip leading list markers for if the model adds "1. "or "- " or something
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)

        if len(line) < 10:
            continue

        idx = line.find(entity)
        if idx == -1:
            Xent.append("")
            Xattr.append(line)
        else:
            char_split = idx + len(entity)
            Xent.append(line[:char_split])
            Xattr.append(line[char_split:])

        if len(Xent) >= SENTENCES_PER_TRIPLET:
            break

    return Xent, Xattr


# =============================================================================
# GET_ATTR — Orchestrator (Algorithm 1, Line 3)
# =============================================================================

def get_attr(model, model_pre, tokenizer, entity: str, language: str = "english", prompt_mode: str = "native") -> tuple[list[str], list[str]]:
    """
    Runs the get_attr process: generates triplets, validates them, and
    converts them to sentences to be split.
    """
    print(f"  [GET_ATTR] Extracting triplets from current model ...")
    raw_triplets = extract_triplets(model, tokenizer, entity, language, prompt_mode)
    print(f"  [GET_ATTR] {len(raw_triplets)} raw triplets extracted.")

    all_Xent, all_Xattr = [], []
    valid = 0

    for triplet in raw_triplets:
        if not validate_triplet(model_pre, tokenizer, triplet, language, prompt_mode):
            continue
        valid += 1
        Xent, Xattr = convert_triplet_to_split_sentences(
            model_pre, tokenizer, triplet, entity, language, prompt_mode
        )
        all_Xent.extend(Xent)
        all_Xattr.extend(Xattr)

    print(f"  [GET_ATTR] {valid} valid triplets -> {len(all_Xent)} sentence pairs.")
    return all_Xent, all_Xattr


# =============================================================================
# L1: Triplet-based loss
# =============================================================================

def compute_l1_loss(model, tokenizer, Xent: list[str], Xattr: list[str]) -> torch.Tensor:
    """
    Gradient ascent on attribution tokens only, so model unlearns facts about entity.
    Tokens from Xent are masked (label=-100) so the loss is computed only over
    Xattr so we can make the model less likely to predict attributions.
    """
    device = next(model.parameters()).device
  
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for ent_text, attr_text in zip(Xent, Xattr):
        if not attr_text.strip():
            continue

        # Encode each part separately so the split is exact
        ent_ids  = tokenizer(ent_text,  return_tensors="pt",
                             add_special_tokens=True).input_ids.to(device)
        attr_ids = tokenizer(attr_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)

        full_ids = torch.cat([ent_ids, attr_ids], dim=1)
        if full_ids.shape[1] > 512:
            full_ids = full_ids[:, :512]

        labels = full_ids.clone()
        labels[0, :ent_ids.shape[1]] = -100 # mask entity tokens from loss

        if (labels != -100).sum() == 0:
            continue

        outputs = model(input_ids=full_ids, labels=labels)
        # we're doing gradient descent with negative values; goal of GD is to make loss smaller, so we just negate it and loss values are actually getting bigger (as in farther from 0).
        # so, since minimizing f(x) = maximizing -f(x), and AdamW is already built to minimize, we'll just make values negative.
        total_loss = total_loss + (-outputs.loss) # minimizing negative loss = maximizing error so model forgets
        count += 1

    if count == 0:
        # No valid pairs — return a zero tensor that still participates in the graph
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


# =============================================================================
# L2: Sentence-based loss
# =============================================================================

def compute_l2_loss(model, tokenizer, Xsent: list[str]) -> torch.Tensor:
    """Normal gradient ascent on the full explanatory sentences."""
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for sent in Xsent:
        if not sent.strip():
            continue

        ids = tokenizer(
            sent, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)

        if ids.shape[1] < 2:
            continue

        outputs = model(input_ids=ids, labels=ids)
        total_loss = total_loss + (-outputs.loss)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


# =============================================================================
# Algorithm 1 — Full loop
# =============================================================================

def run_unlearning(forget_entity: str, language: str = "english", prompt_mode: str = "native", max_epochs: int = MAX_EPOCHS, output_dir: str = "./unlearned_model_output"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ── Load trainable model ──────────────────────────────────────────────────
    tokenizer, model = load_model(MODEL_ID, HF_TOKEN, trainable=True) # this is the model undergoing unlearning

    _, model_pre = load_model(MODEL_ID, HF_TOKEN, trainable=False) # this is the model being used to check whether triplets generated by the unlearned model should be accepted or rejected throughout unlearning
    #model = model.to(device)
    #model_pre = model_pre.to(device)
    
    model_pre.eval()
    for p in model_pre.parameters():
        p.requires_grad = False
    # ── GET_SENT: collect X_sent once with the frozen model ────
    Xsent = get_sent(model_pre, tokenizer, forget_entity, language, prompt_mode)

    # built for doing gradient descent 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_MIN, # lr will be overwritten each epoch via param_groups
    )

    print(f"\nStarting unlearning loop (max {max_epochs} epochs) ...")
    empty_streak = 0

    for epoch in range(1, max_epochs + 1):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()

        # Algorithm 1, Line 3: GET_ATTR on the current model
        Xent, Xattr = get_attr(model, model_pre, tokenizer, forget_entity, language, prompt_mode)

        if not Xent:
            empty_streak += 1
            print(f"  [WARNING] No valid triplets (streak {empty_streak}/{EMPTY_STOP_PATIENCE})")
            if empty_streak >= EMPTY_STOP_PATIENCE:
                print(f"\nNo valid triplets for {EMPTY_STOP_PATIENCE} consecutive epochs. Stopping.")
                break
        else:
            empty_streak = 0

        # Algorithm 1, Line 4: L1 update (triplet-based loss)
        optimizer.zero_grad()
        l1_loss = compute_l1_loss(model, tokenizer, Xent, Xattr)
        l1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Algorithm 1, Line 5: L2 update (sentence-based loss)
        optimizer.zero_grad()
        l2_loss = compute_l2_loss(model, tokenizer, Xsent)
        l2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"lr={lr:.2e} | "
            f"L1={l1_loss.item():.4f} | "
            f"L2={l2_loss.item():.4f} | "
            f"triplet pairs={len(Xent)}"
        )

    # Algorithm 1, Line 7: save unlearned model
    print(f"\nSaving unlearned model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Unlearning")
    parser.add_argument("--entity", type=str, default="Jesus Christ", help="Entity to unlearn")
    parser.add_argument("--language", type=str, choices=["english", "spanish", "patois"], default="english", help="Language for prompts")
    parser.add_argument("--prompt-mode", type=str, choices=["native", "translate"], default="native", help="'native' or 'translate'")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--output-dir", type=str, default="./unlearned_model_output", help="Output directory for model")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: Please set the HF_TOKEN environment variable.")
        print("Example: export HF_TOKEN='your_token_here'")
        exit(1)

    try:
        run_unlearning(args.entity, args.language, args.prompt_mode, args.max_epochs, args.output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")