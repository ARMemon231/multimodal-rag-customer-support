import logging

from app.ingestion import ingest_products

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    count = ingest_products("data/products.json")
    print(f"Ingested {count} chunks")
