import os

from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Environment:
    url: str = os.environ["AZURE_OPENAI_BASE_URL"]
    key: str = os.environ["AZURE_OPENAI_API_KEY"]
    langsmith_key: str = os.environ["LANGSMITH_API_KEY"]

@dataclass
class Models:
    DeepSeek_V4_Flash: str = "DeepSeek-V4-Flash"
    DeepSeek_V4_Pro: str = "DeepSeek-V4-Pro"
    Kimi_K2_6: str = "Kimi-K2.6"
    Mistral_Large_3: str = "Mistral-Large-3"
    Embedding_Small: str = "text-embedding-3-small"
    Gpt_5_4_nano: str = "gpt-5.4-nano"