from uuid import uuid4

from langchain.agents import  create_agent
from langgraph.graph import  StateGraph, MessagesState, START, END
from langchain_openai import  ChatOpenAI
from environ import  Environment, Models
from langgraph.checkpoint.memory import  InMemorySaver
from langchain_core.runnables import  RunnableConfig

llm = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, model=Models.DeepSeek_V4_Flash, temperature=0.4)
checkpoint = InMemorySaver()
config = RunnableConfig(configurable={"thread_id": str(uuid4())})

def prompt_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return  {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(prompt_llm)
builder.add_edge(START, 'prompt_llm')
builder.add_edge('prompt_llm', END)

graph = builder.compile(checkpointer=checkpoint)

while True:
    user_message = str(input("Enter your message: "))
    response = graph.invoke(input={
        "messages": [{
            "role": "user",
            "content": user_message
        }]
    },
    config=config)
    print(response)