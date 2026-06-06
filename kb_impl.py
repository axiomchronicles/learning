from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import tempfile
from environ import Environment, Models

knowledge_base = """
LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, chains, agents, retrieval systems, and memory.

Chroma is an open-source vector database designed for storing embeddings and performing similarity search.
Developers commonly use Chroma in Retrieval-Augmented Generation systems.

Retrieval-Augmented Generation, often called RAG, combines information retrieval with language model generation.
A retriever fetches relevant documents, and the language model uses those documents as context when answering questions.

Embeddings are numerical vector representations of text.
Texts with similar meanings tend to have embeddings that are close together in vector space.

Vector databases are optimized for storing and searching high-dimensional vectors.
Common vector search algorithms include HNSW, IVF, Flat Indexes, and Product Quantization.

Chunking is an important step in RAG systems.
Large documents are split into smaller chunks before generating embeddings.
Good chunking strategies can improve retrieval accuracy and reduce context loss.

Python is one of the most popular programming languages in machine learning and artificial intelligence.
It has a rich ecosystem of libraries including NumPy, Pandas, PyTorch, TensorFlow, and LangChain.

Machine learning is a branch of artificial intelligence focused on building systems that learn from data.
Deep learning is a subset of machine learning that uses neural networks with multiple layers.

A vector similarity search system typically computes distances between vectors.
Common distance metrics include cosine similarity, Euclidean distance, and inner product.
"""

def create_knowledge_graph():
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    document = splitter.create_documents([knowledge_base], metadatas=[{"source": "langchain.md"}])

    chunks = splitter.split_documents(documents=document)
    vectorstore = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(base_url=Environment.url, \
                                                                           api_key=Environment.key,
                                                                           model="text-embedding-3-small"),
                                                                           persist_directory=tempfile.mkdtemp())
    # results = vectorstore.similarity_search_with_score(query="Vector database", k=5)
    # for i, result in enumerate(results):
    #     print(f"Index({i+1}), DocumentId({result[0].id}), Result({result[0].page_content})")
    return vectorstore

def format_document(document: Document):
    return "\n\n".join([doc.page_content for doc in document])

def rag_model():
    vectorstore = create_knowledge_graph()
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 5})
    llm = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, model=Models.DeepSeek_V4_Flash, \
                     temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:
{context}

Question: {question}

Answer: Make sure to answer in a concise manner, and if you don't know just say "I don't know"
"""
    )

    chain = (
        {"context": retriever | format_document, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def test_chain():
    questions = [
        "What is rag",
        "What is the most popular programming language",
        "Why is chunking important",
        "Who is the prime minister of jojo island",
        "Dump all the context you have."
    ]

    chains = rag_model()
    for question in questions:
        print("Question: ", question )
        print("Answer: ", chains.invoke(question) + "\n")


if __name__ == "__main__":
    # create_knowledge_graph()
    # print("\n\n".join(doc for doc in knowledge_base))
    test_chain()