from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_namespace: str = Field(default="products", alias="PINECONE_NAMESPACE")

    top_k: int = Field(default=5, alias="TOP_K")
    min_similarity: float = Field(default=0.35, alias="MIN_SIMILARITY")
    enable_memory: bool = Field(default=True, alias="ENABLE_MEMORY")

    twilio_account_sid: str | None = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: str | None = Field(default=None, alias="TWILIO_WHATSAPP_NUMBER")


settings = Settings()
