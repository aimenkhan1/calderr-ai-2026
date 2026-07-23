""" 
conditional edges practise 
A simple LangGraph-based query routing system that classifies user questions
and directs them to General,Technical,or Sensitive handlers using
conditional graph routing.
"""


from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

#state 
class State(TypedDict):
    query: str          
    category: str      
    response: str        


#node-classification functions for the graph
def classify(state: State) -> dict:
    query = state["query"].lower()

    sensitive_words = ["password", "ssn", "medical", "bank", "salary"]
    technical_words = ["code", "python", "api", "bug", "error", "server", "database"]

    if any(word in query for word in sensitive_words):
        category = "sensitive"
    elif any(word in query for word in technical_words):
        category = "technical"
    else:
        category = "general"

    return {"category": category}


def route_by_category(state: State) -> str:
    return state["category"]


#acc to classification,route to specific handler node
def handle_general(state: State) -> dict:
    return {"response": f"[General Handler] Here's a general answer to: '{state['query']}'"}


def handle_technical(state: State) -> dict:
    return {"response": f"[Technical Handler] Let's debug this: '{state['query']}'"}


def handle_sensitive(state: State) -> dict:
    return {"response": "[Sensitive Handler] This involves sensitive data — routing to a human reviewer."}


#building the graph
builder = StateGraph(State)

builder.add_node("classify", classify)
builder.add_node("handle_general", handle_general)
builder.add_node("handle_technical", handle_technical)
builder.add_node("handle_sensitive", handle_sensitive)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "general": "handle_general",
        "technical": "handle_technical",
        "sensitive": "handle_sensitive",
    },
)

builder.add_edge("handle_general", END)
builder.add_edge("handle_technical", END)
builder.add_edge("handle_sensitive", END)

graph = builder.compile()

#main
if __name__ == "__main__":
    query = input("Enter your question: ").strip()
    result = graph.invoke({"query": query, "category": "", "response": ""})
    print(f"Category: {result['category']}")
    print(result["response"])