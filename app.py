import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import logging

# --- Configuration ---
# USE THE HUB REPOSITORY NAME YOU CREATED IN STEP 1
MODEL_HUB_ID = "naiscriil/youtube_model" # <<< CHANGE THIS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Device will depend on Space hardware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Caching Models (Important for Streamlit Performance) ---
# Use Streamlit's caching to avoid reloading model/tokenizer on every interaction
@st.cache_resource # Use cache_resource for non-serializable objects like models/tokenizers
def load_model_and_tokenizer(model_id):
    logger.info(f"Loading model {model_id}...")
    try:
        # Load model - consider quantization for Spaces (especially CPU/basic GPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit= (DEVICE == "cuda"), # Use 8-bit on GPU if available & bitsandbytes installed
            # device_map="auto" # Use if loading quantized or letting accelerate handle placement
        )
        if DEVICE == "cpu" and torch.cuda.is_available(): # If model loaded on CPU but CUDA exists
             logger.warning("Model loaded on CPU despite CUDA availability (likely due to quantization choice or device_map). Ensure this is intended.")
        elif DEVICE == "cuda" and not hasattr(model, 'hf_device_map'): # Only move if not using device_map
            model.to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set Tokenizer pad_token = eos_token ({tokenizer.eos_token})")

        model.eval() # Set to evaluation mode
        logger.info("Model and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}")
        logger.error(f"Failed loading model/tokenizer: {e}", exc_info=True)
        return None, None

# --- Load Resources ---
model, tokenizer = load_model_and_tokenizer(MODEL_HUB_ID)

# --- (Optional) Stopping Criteria Definition ---
# Define this if you want to stop on newline
class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        if not newline_tokens: raise ValueError("Tokenizer cannot encode newline.")
        self.newline_token_id = newline_tokens[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_len = input_ids.shape[1]
        if current_len <= self.prompt_len: return False
        last_token_ids = input_ids[:, -1]
        found_newline = torch.any(last_token_ids == self.newline_token_id).item()
        return found_newline

# --- Generation Function ---
# Slightly adapted for Streamlit context
def generate_comment_streamlit(
    prompt_text: str,
    max_new_tokens: int = 75,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.9,
    no_repeat_ngram_size: int = 2,
    stop_on_newline: bool = False # Add flag to control stopping
    ) -> str:

    if not model or not tokenizer:
        return "Error: Model or Tokenizer not loaded."

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(model.device if hasattr(model, 'hf_device_map') else DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_token_length = input_ids.shape[1]

        stopping_criteria = None
        if stop_on_newline:
            stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(tokenizer, prompt_token_length)])

        with torch.no_grad():
             # Determine device to use - handle models loaded with device_map
             current_device = model.device if hasattr(model, 'hf_device_map') else DEVICE
             # Use autocast if on GPU for potential speedup
             with torch.autocast(device_type=current_device.split(':')[0], enabled=(current_device != 'cpu')):
                 outputs = model.generate(
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     max_new_tokens=max_new_tokens,
                     do_sample=do_sample,
                     top_k=top_k,
                     top_p=top_p,
                     temperature=temperature,
                     no_repeat_ngram_size=no_repeat_ngram_size,
                     stopping_criteria=stopping_criteria,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id,
                     # Add other generate args if needed
                 )

        generated_ids = outputs[0][prompt_token_length:]
        comment_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop_on_newline and comment_text.endswith('\n'):
             comment_text = comment_text[:-1]

        return comment_text.strip()

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        st.error(f"Generation failed: {e}")
        return "Error during generation."


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("YouTube Comment Generator (Fine-tuned GPT-2 Large)")

if not model or not tokenizer:
    st.error("Model or Tokenizer failed to load. Please check the logs.")
else:
    st.sidebar.header("Video Context")
    title = st.sidebar.text_input("Video Title", "Example: My Trip to the Mountains")
    channel = st.sidebar.text_input("Channel Name", "Example: Adventure Vlogs")
    category = st.sidebar.text_input("Category", "Example: Travel & Events")
    tags_input = st.sidebar.text_input("Tags (comma-separated)", "Example: travel, mountains, hiking")

    st.sidebar.header("Generation Settings")
    max_tokens = st.sidebar.slider("Max New Tokens", 5, 200, 75)
    temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.9, 0.05)
    top_k_val = st.sidebar.slider("Top-K", 0, 100, 50) # 0 disables top-k
    top_p_val = st.sidebar.slider("Top-P (Nucleus)", 0.0, 1.0, 0.95, 0.01) # 0 disables top-p
    use_sampling = st.sidebar.checkbox("Use Sampling", True)
    stop_newline = st.sidebar.checkbox("Stop on First Newline", False)

    if st.button("Generate Comment"):
        if not title or not channel or not category:
            st.warning("Please fill in Title, Channel, and Category.")
        else:
            # Format the prompt for the model
            bos_token_str = tokenizer.bos_token
            separator_str = "\nComment:\n"
            prompt = (
                f"{bos_token_str}\n"
                f"Title: {title}\n"
                f"Channel: {channel}\n"
                f"Category: {category}\n"
                f"Tags: {tags_input if tags_input else 'None'}" # Handle empty tags
                f"{separator_str}"
            )

            with st.spinner("Generating..."):
                generated_comment = generate_comment_streamlit(
                    prompt_text=prompt,
                    max_new_tokens=max_tokens,
                    do_sample=use_sampling,
                    top_k=top_k_val if top_k_val > 0 else None, # Pass None to disable
                    top_p=top_p_val if top_p_val > 0 else None, # Pass None to disable
                    temperature=temp,
                    stop_on_newline=stop_newline
                )
            st.subheader("Generated Comment:")
            st.write(generated_comment)