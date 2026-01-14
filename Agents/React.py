from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage #Foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage #Passes data back to llm after it calls a tool such as the content & the tool_call_id
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class AgentState2(TypedDict):
  mssg:Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a:int,b:int):
  """"this is addtion func that add a+b"""
  return a+b

tools=[add]

def model_call(state:AgentState2)->AgentState2:
  sys_prompt= SystemMessage(conten="You are my Ai assistant please answer my query to best of you abilty")
  res=model.invoke([sys_prompt]+ state["mssg"])
  print("mssg: ",res)
  return {"mssg":[res]}

def should_contniue(state:AgentState2):
  mssg=state["mssg"]
  ls_mssg=mssg[-1]
  if not ls_mssg.tool_calls:
    return "end"
  else:
    return "continue"

graph1=StateGraph(AgentState2)
graph1.add_node("agent",model_call)

tool_node=ToolNode(tools)
graph1.add_node("tool",tool_node)

graph1.add_edge(START,"agent")
graph1.add_conditional_edges(
    "agent",
    should_contniue,
    {
        #edge:node
        "continue":"tool",
        "end":END
    } 
)
graph1.add_edge("tool","agent")
app=graph1.compile()
def print_stream(stream):
    for s in stream:
        message=s["mssg"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()



inputs = {"mssg": [HumanMessage(content="add 5 and 9 and then subtract 4 from the result.")]}
print_stream(app.stream(inputs, stream_mode="values"))