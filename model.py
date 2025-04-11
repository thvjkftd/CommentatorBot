import logging
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.classes.__path__ = []

MODEL_PATH = st.secrets["MODEL_PATH"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=True)
model.to(DEVICE)
model.eval()
logger.info("Loading complete.")

def generate_comment(
    prompt: str,
    max_new_tokens: int = 75,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.9,
    no_repeat_ngram_size: int = 2,
    repetition_penalty: float = 1.0
) -> list[str]:

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_token_length = input_ids.shape[1]

    logger.info("Generating comment...")
    with torch.no_grad():
        with torch.amp.autocast(DEVICE):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    logger.info("Generation complete.")

    generated_id = outputs[0][prompt_token_length:]
    comment_text = tokenizer.decode(generated_id, skip_special_tokens=True)
    generated_comment = comment_text.strip().split('\n')[0]

    return generated_comment
