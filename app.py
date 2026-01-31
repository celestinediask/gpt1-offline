import torch
import os
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

def chat():
    print("Loading GPT-1 model (Offline Mode)...")
    
    # STRICT OFFLINE MODE:
    # We forbid the script from trying to connect to the internet.
    try:
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', local_files_only=True)
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', local_files_only=True)
    except OSError:
        print("\nCRITICAL ERROR: Model not found inside image.")
        print("Did you build the Dockerfile with Layer 3 included?")
        return

    chat_history_ids = None

    print("\n--- GPT-1 Live Chat (Type 'quit' to exit) ---")
    print("Note: GPT-1 is a storyteller. It generates a continuation of your text.\n")

    while True:
        try:
            user_input = input(">> User: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if user_input.lower() in ["quit", "exit"]:
            break

        # Encode input
        new_user_input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Append to history to maintain context
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate response
        outputs = model.generate(
            bot_input_ids, 
            max_length=bot_input_ids.shape[-1] + 50, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95,
            temperature=0.8
        )

        # Decode and print only the new text
        response = tokenizer.decode(outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"GPT-1: {response}")

        # Update history
        chat_history_ids = outputs

if __name__ == "__main__":
    chat()
