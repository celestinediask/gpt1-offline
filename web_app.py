import torch
import gradio as gr
from threading import Thread
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, TextIteratorStreamer

print("Loading GPT-1 model...")
try:
    # Load model once at startup
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', local_files_only=True)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', local_files_only=True)
except Exception as e:
    print(f"Error loading model: {e}")

def chat_stream(message, history):
    # 'history' comes from Gradio as a list of [user_msg, bot_msg] pairs.
    # We can concatenate them for context, but for GPT-1 (which has a small memory),
    # let's just use the current message to keep it simple and fast.
    
    # 1. Prepare Input
    inputs = tokenizer.encode(message, return_tensors='pt')

    # 2. Setup Streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_length=inputs.shape[-1] + 100, # Generate up to 100 new tokens
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    # 3. Generate in background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 4. Stream output to the UI
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# Define the UI
demo = gr.ChatInterface(
    fn=chat_stream,
    title="GPT-1 Offline",
    description="The original 2018 GPT model running locally on your server.",
    examples=["Artificial Intelligence is", "Once upon a time", "The future of space travel"],
    theme="soft"
)

# Launch the server
# server_name="0.0.0.0" allows access from outside the container
demo.launch(server_name="0.0.0.0", server_port=7860)
