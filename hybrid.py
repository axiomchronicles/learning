from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from environ import Models, Environment
import tempfile

document: str = """
# AI Knowledge Base

## LangChain Overview

LangChain is a framework for building applications powered by large language models. It provides tools for prompt management, retrieval pipelines, document loaders, text splitters, agents, memory systems, and evaluation workflows.

Developers frequently use LangChain when building chatbots, question-answering systems, document assistants, and Retrieval-Augmented Generation applications.

---

## Retrieval-Augmented Generation

Retrieval-Augmented Generation, commonly abbreviated as RAG, combines document retrieval with language model generation.

A retriever searches a knowledge base and returns relevant documents. The language model then uses those retrieved documents as context to generate accurate responses.

RAG systems help reduce hallucinations and improve factual grounding.

---

## Vector Databases

Vector databases store embeddings and enable similarity search across large collections of documents.

Popular vector databases include Chroma, Pinecone, Weaviate, Milvus, and Qdrant.

Vector search works by comparing numerical representations of text rather than exact keywords.

---

## Chroma Database

Chroma is an open-source embedding database designed for AI applications.

It supports document storage, metadata filtering, semantic search, similarity search, and persistence to local disk.

Many developers choose Chroma for local RAG experiments because it is lightweight and easy to integrate with LangChain.

---

## Embeddings

Embeddings transform text into dense numerical vectors.

Two sentences may use different words but still produce similar embeddings if they have similar meanings.

For example:

"The automobile accelerated rapidly."

and

"The car sped up quickly."

contain different vocabulary but similar semantic meaning.

---

## Keyword Search

Keyword search retrieves documents by matching exact terms.

A lexical retrieval system such as BM25 may rank documents highly when the query contains words that exactly appear in the document.

For example, a query containing "vector database" strongly matches documents that explicitly contain those terms.

---

## Semantic Search

Semantic search focuses on meaning rather than exact wording.

A user searching for "AI document lookup" may still retrieve documents discussing Retrieval-Augmented Generation even if the exact phrase is not present.

Semantic retrieval is powered by embeddings and vector similarity calculations.

---

## Hybrid Retrieval

Hybrid retrieval combines lexical search and semantic search.

BM25 provides strong keyword matching while vector retrieval captures semantic relationships.

An ensemble retriever often produces better results than either retrieval strategy alone.

---

## Python for AI

Python is one of the most widely used programming languages for artificial intelligence and machine learning.

Popular libraries include NumPy, Pandas, PyTorch, TensorFlow, and LangChain.

Python's ecosystem makes it a common choice for building retrieval systems and vector search applications.

---

## Machine Learning

Machine learning is a field of artificial intelligence that enables systems to learn patterns from data.

Deep learning is a specialized branch of machine learning that uses neural networks with many layers.

Large language models are a modern application of deep learning.

---

## Synonym Test Section

A vehicle can also be called an automobile or a car.

A database may be described as a datastore, storage engine, or persistence layer.

An AI assistant may also be referred to as a chatbot, virtual assistant, conversational agent, or intelligent helper.

These alternative terms are useful when evaluating semantic retrieval systems.
"""


def split_markdown():
    splitter = MarkdownTextSplitter(chunk_size = 500, chunk_overlap = 50)
    documents = splitter.create_documents([document])
    print(f"DEBUG: {len(documents[0].page_content)}")
    return documents


def begin_retriever():
    chunk = split_markdown()
    vectorstore = Chroma.from_documents(chunk, OpenAIEmbeddings(base_url=Environment.url, api_key=Environment.key, \
                                                                model=Models.Embedding_Small), persist_directory=tempfile.mkdtemp())
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(documents=chunk)

    ensemble = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.5, 0.5])
    return ensemble

def main():
    questions: list[str] = [
        "What is chroma?",
        "Explain BM25 retrieval.",
        "What is a vector database?",
        "What software stores embedding vectors?",
        "How can an AI system answer questions using external documents?",
        "Tell me about automobiles.",
        "Tell me about cars.",
        "Which local datastore can be used for semantic search?",
        "How do chatbots use retrieved knowledge to generate grounded responses?"
    ]

    chain = begin_retriever()
    for i, question in enumerate(questions):
        answer = chain.invoke(question)
        print(f"Q{i+1}. {question}")
        print(f"Ans {answer} \n")


if __name__ == "__main__":
    main()