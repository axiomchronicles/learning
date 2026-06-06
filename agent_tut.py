from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel, Field
from environ import Environment, Models

import typing
import requests

model: ChatOpenAI = ChatOpenAI(base_url=Environment.url, api_key=Environment.key, model=Models.DeepSeek_V4_Flash, \
                               temperature=0.3)
checkpoint: InMemorySaver = InMemorySaver()

# header: typing.Dict[typing.List[str, typing.Any], typing.Any] = {
#     ["New York", "UN"], {"temp": ["fahrenheit", "celsius"]}
# }

class Context(BaseModel):
    user_id: str = Field(...)

@tool("get_weather", description="Get the weather for the current city", return_direct=False)
def get_weather(city: str) -> typing.Dict[str, typing.Any]:
    response = requests.get(url=f"https://wttr.in/{city}?format=j1")
    return response.json()

@tool("get_user", description="Get user's location by id", return_direct=False)
def get_user(runtime: ToolRuntime[Context]) -> str:
    user_id: str = runtime.context.user_id
    match user_id:
        case "DDX1":
            return "Faridabad"
        case "DDX2":
            return "Patna"
        case "DDX3":
            return "Amritsar"
        case _:
            return "Unknown"

class ResponseFormat(BaseModel):
    summary: str = Field(..., description="Weather Summary")
    fahrenheit: float = Field(..., description="Weather temperature in Fahrenheit")
    celsius: float = Field(..., description="Weather temperature in Celsius")
    humidity: float = Field(..., description="Weather humidity")


agent = create_agent(
    model=model, tools=[get_weather, get_user],system_prompt="""
You are a weather assistant.

When asked for weather:

1. Use get_user to determine the user's city.
2. Use get_weather with that city.
3. Produce a ResponseFormat object.

If the user cannot be identified, say you don't know.
""", response_format=ResponseFormat, context_schema=Context, checkpointer=checkpoint
)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ]
    }, config=config, context=Context(user_id="DDX1")
)
print(response["structured_response"].summary)

