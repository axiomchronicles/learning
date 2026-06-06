from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader, PyPDFLoader

from pathlib import Path, PurePath

import os
import tempfile
import typing

load_dotenv()

def func() -> str:
    @dataclass
    class Models:
        DeepSeek_V4_Flash: str = "DeepSeek-V4-Flash"
        DeepSeek_V4_Pro: str = "DeepSeek-V4-Pro"
        Kimi_K2_6: str = "Kimi-K2.6"


    model = ChatOpenAI(
        base_url = os.environ["AZURE_OPENAI_BASE_URL"],
        api_key = os.environ["AZURE_OPENAI_API_KEY"],
        model = Models.DeepSeek_V4_Flash,
        temperature = 0
    )

    message = model.invoke(input="Setup Completed!")
    print(message.text)


def create_and_load_docs() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"Hi, This is a temperory file. \n Create by using the python-tempfile. \n have no real purpose just to test the langchain TextLoader.")
        tmp.flush()
        tmp_file_path: str = tmp.name

        try:
            loader = TextLoader(tmp_file_path)
            content = loader.load()
            for doc in content:
                print(doc.page_content)

        finally:
            os.remove(tmp_file_path)


def pdf_loader():
    loader = PyPDFLoader(file_path = PurePath("./docs/test.pdf"))
    content = loader.load()

    print(content[0].page_content)


# def func_embedding()

if __name__ == "__main__":
    # create_and_load_docs()
    # pdf_loader()
    func()