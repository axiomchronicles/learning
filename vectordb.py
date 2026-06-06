from chromadb import Client
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import tempfile

from environ import Environment

client = Client()

collection = client.get_or_create_collection(name="mycollection")

docs = [
    Document(page_content="LangChain is a framework for LLM applications."),
    Document(page_content="Chroma is a vector database."),
    Document(page_content="This is a fake context")
]

def func() -> None:
    for doc in docs:
        query = collection.add(ids=doc["id"], documents=doc["context"])
        print("Added (+): ", query)

    query = collection.query(query_texts="Hello World!")
    print("Found: ", query)


def similarity_search() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        vectorstore = Chroma.from_documents(documents=docs, \
                                            embedding=OpenAIEmbeddings(base_url=Environment.url, \
                                                                       api_key=Environment.key, model="text-embedding-3-small"),
                                            persist_directory=tempdir)
        
        print(f"Vectorstore create: Total Document: {vectorstore._collection.count()}")

        query = "Fake Context"
        # similarity_search = vectorstore.similarity_search(query=query, k=3)
        # print("Similarity Result: ", similarity_search)

        similarity_search_score = vectorstore.similarity_search_with_score(query=query, k=3)
        print(similarity_search_score)

if __name__ == "__main__":
    similarity_search()