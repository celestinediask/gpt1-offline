# Base image
FROM python:3.9-slim

WORKDIR /app

# --- LAYER 1: Core Dependencies (CPU Optimized) ---
COPY requirements.txt .

# 1. Install CPU-only PyTorch (This saves ~2GB of space!)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest (transformers)
RUN pip install --no-cache-dir -r requirements.txt

# --- LAYER 2: Compile Optimization Libs ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir ftfy spacy && \
    python -m spacy download en_core_web_sm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- LAYER 3: Bake Model into Image ---
# (This still works the same, baking the model file inside)
RUN python -c "from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel; \
    OpenAIGPTTokenizer.from_pretrained('openai-gpt'); \
    OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')"

# --- LAYER 4: App Code ---
COPY app.py .

CMD ["python", "-u", "app.py"]
