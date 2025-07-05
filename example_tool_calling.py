from typing import Annotated, Sequence, TypedDict, Union
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from utils import print_stream

load_dotenv()

Number = Union[float, int]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
builder = StateGraph(AgentState)

# tools
@tool
def multiply(a: Number, b: Number) -> Number:
    """Tool use to multiply two numbers in parameters a and b"""
    return a * b
@tool
def add(a: Number, b: Number) -> Number:
    """Tool use to add two numbers in parameters a and b"""
    return a + b
tools = [multiply, add]

# model
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm = llm.bind_tools(tools)

# nodes
def should_call_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    
    if last_msg.tool_calls:
        return "call"
    else:
        return "end"
    
def agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}
tool_node = ToolNode(tools)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# connections
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", should_call_tools,
    {"call": "tools", "end": END}
)
builder.add_edge("tools", "agent")
app = builder.compile()

inputs = {"messages": [("user", "Multiply 5 by 7 then add 3")]}
print_stream(app.stream(inputs, stream_mode="values"))

# # save graph diagram
# try:
#     with open("graph_tool_calling.png", "wb") as f:
#         f.write(app.get_graph().draw_mermaid_png())
# except Exception:
#     print("Not possible to generate graph diagram")






