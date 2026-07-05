import logging

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from app.config import settings
from app.memory import ConversationMemory
from app.retrieval import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal RAG Customer Support")
rag = RAGService()
memory = ConversationMemory()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running", "message": "Multimodal RAG Customer Support is live. Use POST /webhook/whatsapp"}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/whatsapp", response_class=PlainTextResponse)
def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(default=""),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str = Form(default=""),
) -> PlainTextResponse:
    response = MessagingResponse()

    is_image = int(NumMedia) > 0 and MediaUrl0
    if is_image:
        result = rag.answer_image(MediaUrl0, user_question=Body or "What product is this?")
    else:
        ctx = memory.get_context(From) if settings.enable_memory else ""
        result = rag.answer_text(Body, memory_context=ctx)

    if settings.enable_memory:
        memory.add_turn(From, f"User: {Body}")
        memory.add_turn(From, f"Assistant: {result.answer}")

    response.message(result.answer)
    logger.info("user=%s image=%s sources=%s", From, is_image, len(result.sources))
    return PlainTextResponse(content=str(response), media_type="application/xml")
