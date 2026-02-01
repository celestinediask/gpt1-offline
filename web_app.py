import torch
import gradio as gr
from threading import Thread
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, TextIteratorStreamer

# --- CONFIGURATION ---
MODEL_NAME = 'openai-gpt'
MAX_MODEL_TOKENS = 512

print(f"Loading {MODEL_NAME} model...")
try:
    tokenizer = OpenAIGPTTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = OpenAIGPTLMHeadModel.from_pretrained(MODEL_NAME, local_files_only=True)

    # --- THE CRASH FIX: EMERGENCY TOKEN ---
    # The original GPT-1 has no "End of Text" token, which causes the NoneType error.
    # We manually add one so the code can function.
    if tokenizer.eos_token is None:
        print("Adding missing EOS token to prevent crash...")
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        tokenizer.pad_token = tokenizer.eos_token
        # We must resize the model to accept this new vocabulary word
        model.resize_token_embeddings(len(tokenizer))

except Exception as e:
    print(f"CRITICAL ERROR: {e}")

def chat_stream(message, history, max_new_tokens, temperature, top_k):
    # 1. Build Conversation History
    context_str = ""
    for user_msg, bot_msg in history:
        # Now tokenizer.eos_token is guaranteed to be a string, not None
        context_str += user_msg + tokenizer.eos_token + bot_msg + tokenizer.eos_token
    
    context_str += message + tokenizer.eos_token
    
    # 2. Tokenize
    input_ids = tokenizer.encode(context_str, return_tensors='pt')

    # 3. Sliding Window (Fix for 512 limit)
    allowed_input_len = MAX_MODEL_TOKENS - max_new_tokens
    if input_ids.shape[-1] > allowed_input_len:
        input_ids = input_ids[:, -allowed_input_len:]
        print(f"Trimming history to last {allowed_input_len} tokens.")

    # 4. Setup Streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=input_ids,
        streamer=streamer,
        max_length=input_ids.shape[-1] + max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=0.95,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# --- THE UI ---
demo = gr.ChatInterface(
    fn=chat_stream,
    title="GPT-1 (2018) Research Lab",
    description=f"Authentic OpenAI GPT-1 Model | {MAX_MODEL_TOKENS} Token Limit | Offline Mode",
    theme="soft",
       
    additional_inputs=[
        gr.Slider(10, 200, value=50, label="Max New Tokens (Length)"),
        gr.Slider(0.1, 1.5, value=0.8, label="Temperature (Creativity)"),
        gr.Slider(1, 100, value=40, step=1, label="Top-K (Word Filter)"),
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
