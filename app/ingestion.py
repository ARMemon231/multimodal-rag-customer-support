import json
import logging
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.models import ProductDoc

logger = logging.getLogger(__name__)


def _ensure_index(pc: Pinecone) -> None:
    existing_idx = None
    for idx in pc.list_indexes():
        name = idx.name if hasattr(idx, "name") else idx.get("name")
        if name == settings.pinecone_index_name:
            existing_idx = idx
            break

    target_dimension = 384
    if existing_idx is not None:
        dim = existing_idx.dimension if hasattr(existing_idx, "dimension") else existing_idx.get("dimension")
        if dim == target_dimension:
            return
        logger.info("Deleting existing Pinecone index '%s' due to dimension mismatch (found %s, expected %s).", settings.pinecone_index_name, dim, target_dimension)
        pc.delete_index(settings.pinecone_index_name)

    logger.info("Creating Pinecone index '%s' with dimension %s...", settings.pinecone_index_name, target_dimension)
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=target_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )


def load_products(path: str) -> list[ProductDoc]:
    payload = json.loads(Path(path).read_text())
    return [ProductDoc(**item) for item in payload]


def ingest_products(path: str) -> int:
    products = load_products(path)

    docs: list[Document] = []
    for p in products:
        text = (
            f"Product: {p.product_name}\n"
            f"Category: {p.category}\n"
            f"Price: ${p.price:.2f}\n"
            f"Description: {p.description}\n"
            f"Tags: {', '.join(p.tags)}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "price": p.price,
                    "category": p.category,
                    "tags": p.tags,
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    _ensure_index(pc)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    PineconeVectorStore(index=index, embedding=embeddings, namespace=settings.pinecone_namespace).add_documents(chunks)

    logger.info("Ingested %s chunks", len(chunks))
    return len(chunks)
