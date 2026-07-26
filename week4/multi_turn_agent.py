"""
Multi-turn Agent with Accumulating State
Demonstrates TypedDict + Annotated reducers for state that grows over turns.
"""

from typing_extensions import TypedDict
from typing import List, Annotated
import operator
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str                                  
    messages: Annotated[List[str], operator.add]    
    tool_calls: Annotated[List[str], operator.add]      
    intermediate_results: Annotated[List[str], operator.add]  
    turn: int                                          


def receive_message(state: State) -> dict:
    turn = state["turn"] + 1
    return {
        "messages": [f"User (turn {turn}): {state['user_input']}"],  
        "turn": turn,
    }


def decide_tool(state: State) -> dict:
    query = state["user_input"].lower()
    if "weather" in query:
        tool = "weather_api"
    elif "search" in query:
        tool = "web_search"
    else:
        tool = "none"
    return {"tool_calls": [f"Turn {state['turn']}: called '{tool}'"]}


def run_tool(state: State) -> dict:
    result = f"Turn {state['turn']}: tool returned some placeholder result"
    return {"intermediate_results": [result]}


def respond(state: State) -> dict:
    reply = f"Assistant (turn {state['turn']}): here's my answer based on what I found."
    return {"messages": [reply]}  


builder = StateGraph(State)

builder.add_node("receive_message", receive_message)
builder.add_node("decide_tool", decide_tool)
builder.add_node("run_tool", run_tool)
builder.add_node("respond", respond)

builder.add_edge(START, "receive_message")
builder.add_edge("receive_message", "decide_tool")
builder.add_edge("decide_tool", "run_tool")
builder.add_edge("run_tool", "respond")
builder.add_edge("respond", END)

graph = builder.compile()



if __name__ == "__main__":
    state = {
        "user_input": "",
        "messages": [],
        "tool_calls": [],
        "intermediate_results": [],
        "turn": 0,
    }

#running the 3 iterations and each time sending the accumulated state to the graph

    for _ in range(3):   #
        user_input = input("\nYou: ").strip()
        state["user_input"] = user_input
        state = graph.invoke(state)   

    print("\nFull Conversation History")
    for m in state["messages"]:
        print(m)

    print("\nTool Calls Made")
    for t in state["tool_calls"]:
        print(t)

    print("\nIntermediate Results")
    for r in state["intermediate_results"]:
        print(r)