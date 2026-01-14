from typing import TypedDict,List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv

class AgentState2(TypedDict):
  mssg:List[Union[HumanMessage,AIMessage]]
  #ai_mssg:List[AIMessage]

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process(state:AgentState2)->AgentState2:
  res=llm.invoke(state["mssg"])
  state["mssg"].append(AIMessage(content=res.content))
  print(f"AI: {res.content}")
  return state

graph1=StateGraph(AgentState2)
graph1.add_node("process",process)
graph1.add_edge(START,"process")
graph1.add_edge("process",END)
app=graph1.compile()
convo_history=[]
user_input=input("User: ")
while user_input!="exit":
    convo_history.append(HumanMessage(content=user_input))
    response=app.invoke({"mssg":convo_history})
    print(response["mssg"])
    convo_history=response["mssg"]
    user_input=input("User: ")