from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url = os.environ["AZURE_OPENAI_BASE_URL"],
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
)


emb = client.embeddings.create(model = "text-embedding-3-small", input="Hi, my name is pawan")
print(emb)


