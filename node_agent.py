from ctypes.wintypes import WORD
from uuid import uuid4
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import TypedDict
from hybrid import document as hybrid_document
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.tools import tool, ToolRuntime, BaseTool
from langgraph.prebuilt import  ToolNode
from langgraph.types import interrupt, Command

from environ import Environment, Models
from pydantic import BaseModel, Field
from pathlib import Path

import tempfile
import typing

class BaseToolkit:
    WORKSPACE: Path = Path(__file__).cwd()

    @staticmethod
    def safe_path(path: str) -> ValueError | Path:
        p: Path = (BaseToolkit.WORKSPACE / path).resolve()
        if not str(p).startswith(str(BaseToolkit.WORKSPACE)):
            return ValueError("Path outside workspace")
        return p

    @staticmethod
    @tool("read", description="Read the contents of a file from the given path.")
    def read(path: str ) -> str:
        safe_path = BaseToolkit.safe_path(path)
        with open(safe_path) as f:
            return f.read()

    @staticmethod
    @tool("write", description="Write content to a file")
    def write(path: str, content: str = "") -> str:
        safe_path = BaseToolkit.safe_path(path)
        try:
            with open(safe_path, "w") as f:
                f.write(content)
            return  "File written"
        except Exception as e:
            return str(e)

    @staticmethod
    @tool("edit", description="Replace text inside a file.")
    def edit(path: str, old_content: str = "", new_content: str = "") -> str:
        safe_path = BaseToolkit.safe_path(path)
        try:
            with open(safe_path, "r") as f:
                content = f.read()

            content = content.replace(old_content, new_content)
            with open(path, "w") as f:
                f.write(content)

            return "File Updated Successfully"
        except Exception as e:
            return str(e)

toolkit: typing.List[BaseTool] = [BaseToolkit.read, BaseToolkit.write, BaseToolkit.edit]
llm: ChatOpenAI = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, \
                             model=Models.Gpt_5_4_nano, temperature=0.3)
coding_llm: ChatOpenAI = llm.bind_tools(tools=toolkit)
embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(base_url=Environment.url, api_key=Environment.key, \
                                                     model=Models.Embedding_Small)
checkpoint: InMemorySaver = InMemorySaver()
config: RunnableConfig = RunnableConfig(configurable={"thread_id": str(uuid4())})
tool_node: ToolNode = ToolNode(tools=toolkit)


def split_text(text: str):
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    document = splitter.create_documents(texts=[text])
    return document


def create_ensembler(document: list[Document]) -> EnsembleRetriever:
    vector_store = Chroma.from_documents(document, embedding=embedding_model, persist_directory=tempfile.gettempdir())
    semantic_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(document)

    ensembler = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.5, 0.5])
    return ensembler


class IntentClassifier(BaseModel):
    message_intent: typing.Literal["chat", "knowledge_base", "code"] = Field(..., description="Intent Classifier")


class State(TypedDict):
    messages: typing.Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None

def accept_coding(state: State):
    content = state['messages'][-1].content
    decision = interrupt(value="About to run tool write/edit \n\n %s\n\nApprove (yes/no or revised)?" % content)
    text: str = str(decision).strip().lower()
    if text in ["yes", "y", "Y", "Approve"]:
        return {"next_node": "code_agent"}
    elif text in ["no", "n", "N", "Denied"]:
        return {"messages": [AIMessage(content="Request is denied by user")], "next_node": "denied"}

    return {"messages": [HumanMessage(content=text)], "next_node": "accept_coding"}

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    response = structured_llm.invoke(input=[
        SystemMessage(
            "Classify weather the user wants to (chat) or retrieve knowledge (kb) or want to write/edit (code)"),
        HumanMessage(state["messages"][-1].content)
    ])
    return {"message_intent": response.message_intent}


def route_tool(state: State):
    query = state["messages"][-1]

    if getattr(query, "tool_calls", None):
        return "tools"

    return END

def prompt_llm_chat(state: State):
    response = llm.invoke(input=[
        SystemMessage("Your a playful and cheerful talkative chat agent"), *state["messages"]
    ])
    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }


def prompt_llm_rag(state: State):
    query = state["messages"][-1].content
    chunks: list[Document] = split_text(hybrid_document)
    retriever = create_ensembler(document=chunks)

    docs: typing.Any = retriever.invoke(input=query, config=config)
    context: typing.Any = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(input=[
        SystemMessage(
        f"""
You are a retrieval assistant.

Answer ONLY using the retrieved context.

If the answer cannot be found in the context,
say "I could not find that information."

Retrieved Context:
{context}
        """
        ), HumanMessage(query)
    ])
    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }


def prompt_llm_code(state: State):
    response = coding_llm.invoke(input=[
        SystemMessage("""
You are an expert software engineer.

Use tools when necessary.

You may:
- read files
- write files
- inspect code

Always explain your actions.
        """), *state["messages"]
    ])
    return {
        "messages": [response]
    }


graph = StateGraph(State)
graph.add_node("classifier", classify_intent)
graph.add_node("chat_agent", prompt_llm_chat)
graph.add_node("rag_agent", prompt_llm_rag)
graph.add_node("code_agent", prompt_llm_code)
graph.add_node("tools", tool_node)
graph.add_node("accept_coding", accept_coding)

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", lambda state: state["message_intent"], {
    "chat": "chat_agent",
    "knowledge_base": "rag_agent",
    "code": "accept_coding"
})
graph.add_conditional_edges("accept_coding", lambda state: "end" if state.get("next_node") == "denied" \
                            else state["next_node"], {"end": END, "code_agent": "code_agent",
                                                          "accept_coding": "accept_coding"})
graph.add_conditional_edges("code_agent", route_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "code_agent")
graph.add_edge("chat_agent", END)
graph.add_edge("rag_agent", END)
graph.add_edge("code_agent", END)

compiled = graph.compile(checkpointer=checkpoint)
compiled.get_graph().draw_mermaid_png(output_file_path="output.png")

while True:
    user_input = str(input("Enter your message: "))
    response = compiled.invoke(input={
        "messages": [
            HumanMessage(user_input)
        ]
    }, config=config)

    while "__interrupt__" in response:
        prompt = response["__interrupt__"][0].value
        decision = input(f"{prompt}\n>")
        response = compiled.invoke(Command(resume=decision), config=config)
    print(response["messages"][-1].content)
