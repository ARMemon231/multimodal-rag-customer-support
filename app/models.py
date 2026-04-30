from pydantic import BaseModel


class ProductDoc(BaseModel):
    product_id: str
    product_name: str
    description: str
    price: float
    category: str
    tags: list[str]


class RAGResult(BaseModel):
    answer: str
    sources: list[dict]
