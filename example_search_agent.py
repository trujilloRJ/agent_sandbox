import json
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]

tool = TavilySearch(max_results=2)
tools = [tool]

graph_builder = StateGraph(State)

llm = init_chat_model(model="google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# chatbot node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
graph_builder.add_node("chatbot", chatbot)

# tool node
class CustomToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            msg = messages[-1]
        else:
            raise ValueError("No messages fround")
        outputs = []
        for tool_call in msg.tool_calls:
            tool_name = tool_call["name"]
            res = self.tools_by_name[tool_name].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(res),
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                )
            )
            return {"messages": outputs}
tool_node = CustomToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


# conneting graph
def route_tools(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    source="chatbot", path=route_tools,
    path_map={"tools": "tools", END: END} # this is the default path map
)
graph_builder.add_edge("tools", "chatbot") # return path
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break