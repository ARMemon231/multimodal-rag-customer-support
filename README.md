# Multimodal RAG Customer Support (WhatsApp + Gemini + Pinecone)

Production-ready starter for a WhatsApp customer-support agent with:
- Text RAG over product KB
- Image-to-product retrieval using Gemini Vision + Pinecone
- FastAPI webhook for Twilio WhatsApp integration

## Features
- LangChain-based retrieval pipelines
- Separate text and image query chains
- Strict grounded answering from retrieved context
- Fallback handling when retrieval confidence is low
- Optional conversation memory
- Structured logging

## Project Structure
- `app/config.py` - environment/config management
- `app/models.py` - pydantic request/response models
- `app/ingestion.py` - product KB ingestion to Pinecone
- `app/retrieval.py` - text + image RAG pipelines
- `app/webhook.py` - FastAPI + Twilio webhook routes
- `app/memory.py` - in-memory chat history store
- `data/products.json` - sample KB
- `scripts/run_ingestion.py` - load sample data

## Requirements
- Python 3.10+
- Pinecone index
- Google Gemini API key
- Twilio WhatsApp sandbox or production sender

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=products-support
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=products
TOP_K=5
MIN_SIMILARITY=0.35
ENABLE_MEMORY=true
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

## Ingest sample product data
```bash
python scripts/run_ingestion.py
```

## Run API
```bash
uvicorn app.webhook:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 Ngrok Setup (Local Tunnel)

Twilio needs a public HTTPS URL — Ngrok creates a tunnel to your localhost.

### Install Ngrok
Download: https://ngrok.com/download

### Setup Auth Token
1. Sign up: https://dashboard.ngrok.com/signup
2. Get token: https://dashboard.ngrok.com/get-started/your-authtoken
3. Run:
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Start Tunnel
```bash
ngrok http 8000
```
Copy the URL: `https://xxxx-xxxx.ngrok-free.dev`

Configure Twilio webhook to:
`POST https://xxxx-xxxx.ngrok-free.dev/webhook/whatsapp`

## Notes for production
- Use a durable memory backend (Redis) instead of in-memory store.
- Add request signature validation for Twilio.
- Add retry queues for media fetch and LLM calls.
- Deploy behind HTTPS reverse proxy.
