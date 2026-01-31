# Base image
FROM python:3.9-slim

WORKDIR /app

# --- LAYER 1: Core Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- LAYER 2: Compile Optimization Libs ---
# Installs gcc (build-essential) to compile spacy/ftfy
# This fixes the "BasicTokenizer" warning
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir ftfy spacy && \
    python -m spacy download en_core_web_sm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- LAYER 3: Bake Model into Image (Offline Fix) ---
# Downloads the ~500MB model now so it is saved inside the image.
RUN python -c "from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel; \
    OpenAIGPTTokenizer.from_pretrained('openai-gpt'); \
    OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')"

# --- LAYER 4: App Code ---
COPY app.py .

# Run unbuffered so logs appear instantly
CMD ["python", "-u", "app.py"]
