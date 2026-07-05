import logging
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config import settings
from app.models import RAGResult

logger = logging.getLogger(__name__)

TEXT_PROMPT = ChatPromptTemplate.from_template(
    """You are a customer support assistant. Use only the context below.
If the answer is not in context, say: "I couldn't find that in our product catalog.".
Keep answer concise and helpful.

Context:
{context}

Question:
{question}
"""
)


class RAGService:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        self.vectorstore = PineconeVectorStore(index=index, embedding=self.embeddings, namespace=settings.pinecone_namespace)

        self.chat_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=settings.google_api_key, temperature=0.1)
        self.vision_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=settings.google_api_key, temperature=0.1)

    def _retrieve(self, query: str) -> list[Any]:
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=settings.top_k, namespace=settings.pinecone_namespace
        )

    def answer_text(self, question: str, memory_context: str = "") -> RAGResult:
        hits = self._retrieve(question)
        filtered = [h for h in hits if h[1] >= settings.min_similarity]
        if not filtered:
            logger.warning("No relevant docs found in Pinecone for: %s", question)
            return RAGResult(answer="I'm sorry, I don't have information about this product in our catalog. Please contact our support team for further assistance.", sources=[])

        context = "\n\n".join(doc.page_content for doc, _ in filtered)
        if memory_context:
            context = f"Conversation history:\n{memory_context}\n\n{context}"

        prompt = TEXT_PROMPT.format_prompt(context=context, question=question)
        answer = self.chat_llm.invoke(prompt.to_messages()).content
        sources = [doc.metadata for doc, _ in filtered]
        return RAGResult(answer=answer, sources=sources)

    def answer_image(self, image_url: str, user_question: str = "Identify this product") -> RAGResult:
        vision_prompt = [
            {
                "type": "text",
                "text": "Extract a concise product description including product type, color, materials, and likely use case.",
            },
            {"type": "image_url", "image_url": image_url},
        ]
        description = self.vision_llm.invoke([{"role": "user", "content": vision_prompt}]).content
        query = f"{user_question}. Visual description: {description}"
        return self.answer_text(query)
