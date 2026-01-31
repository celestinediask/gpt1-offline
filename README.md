# GPT-1 Offline Container

A fully self-hosted, offline-capable Docker container for the original GPT-1 model (2018).

## Features
- **Zero Internet Required:** Model is baked into the Docker image.
- **Privacy First:** "Phone home" features disabled.
- **Easy Deploy:** One command setup via Docker Compose.

## How to Run
1. Build the image:
   `docker compose build`
2. Run the chat:
   `docker compose run --rm gpt-runner`
