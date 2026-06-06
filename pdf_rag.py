from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from environ import Environment, Models
from pathlib import PurePath
import tempfile

class PdfRagChain:
    def __init__(self):
        pass

    def load_document(self):
        loader = PyPDFLoader(file_path=PurePath("./docs/test.pdf"))
        loaded_documents = loader.load()
        return loaded_documents
    
    def create_knowledeg_graph(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        documents = self.load_document()

        chunker = splitter.split_documents(documents=documents)
        vectorstore = Chroma.from_documents(documents=chunker, embedding=OpenAIEmbeddings(base_url=Environment.url, \
                                                                                          api_key=Environment.key, model=Models.Embedding_Small),
                                            persist_directory=tempfile.mkdtemp())
        return vectorstore
    
    def format_document(self, document: Document):
        return "\n\n".join([doc.page_content for doc in document])
    
    def rag_chain(self):
        vectorstore = self.create_knowledeg_graph()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        llm = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, model=Models.DeepSeek_V4_Flash, \
                         temperature=0.4)
        prompt = ChatPromptTemplate.from_template(
            """
Answer the question based on the following context:
{context}

Question: {question}

Answer: Make sure to answer in a concise manner. If you are not sure then just say "I don't know"
"""
        )

        chain = (
            {"context": retriever | self.format_document, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain
    
    def ask_question(self):
        while True:
            question: str = str(input("Enter your question: "))
            chain = self.rag_chain()
            response = chain.invoke(question)
            print(response)
        
if __name__ == "__main__":
    print(PdfRagChain().ask_question())