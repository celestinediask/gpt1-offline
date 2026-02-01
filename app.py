import torch
import os
from threading import Thread
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, TextIteratorStreamer

def chat():
    print("Loading GPT-1 model (Offline Mode)...")
    
    try:
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', local_files_only=True)
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', local_files_only=True)
    except OSError:
        print("\nCRITICAL ERROR: Model not found inside image.")
        return

    # Use a simpler conversation history for streaming
    chat_history_ids = None

    print("\n--- GPT-1 Live Chat (Streaming Enabled) ---")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("\n>> User: ")
        except KeyboardInterrupt:
            break
        
        if user_input.lower() in ["quit", "exit"]:
            break

        # 1. Prepare Input
        new_user_input_ids = tokenizer.encode(user_input, return_tensors='pt')
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # 2. Setup Streaming
        # The streamer intercepts the text as it is generated
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs=bot_input_ids,
            streamer=streamer,
            max_length=bot_input_ids.shape[-1] + 50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        # 3. Generate in a background thread
        # We need a thread because .generate() is "blocking" (it waits until done)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 4. Print tokens as they arrive (The "Instant" feel)
        print("GPT-1: ", end="", flush=True)
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print() # Newline after finishing

        # 5. Update history (Re-encode the full text to keep history simple)
        # This is a bit inefficient but safe for this script
        full_text = user_input + generated_text
        chat_history_ids = tokenizer.encode(full_text, return_tensors='pt')

if __name__ == "__main__":
    chat()
