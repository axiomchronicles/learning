from langsmith import traceable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from environ import Environment, Models
import os

# os.environ["LANGSMITH_TRACING"] = True
load_dotenv()

@traceable(name="My Test Function", tags=["openai", "test"])
def main():
    llm = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, model=Models.DeepSeek_V4_Flash, temperature=0.4)
    return llm.invoke("Write a basic fastapi code")


if __name__ == "__main__":
    main()