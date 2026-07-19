from typing_extensions import TypedDict          
from langgraph.graph import StateGraph, START, END  

class State(TypedDict):                            
    name: str                                      
    greeting: str                                 
    shout: str                                    
    final: str                                    

def say_hello(state: State) -> dict:                
    return {"greeting": f"Hello, {state['name']}!"}  

def make_loud(state: State) -> dict:               
    return {"shout": state["greeting"].upper()}     

def add_signature(state: State) -> dict:             
    return {"final": f"{state['shout']} — from LangGraph "}  

builder = StateGraph(State)                        

builder.add_node("say_hello", say_hello)          
builder.add_node("make_loud", make_loud)            
builder.add_node("add_signature", add_signature)  

builder.add_edge(START, "say_hello")                
builder.add_edge("say_hello", "make_loud")           
builder.add_edge("make_loud", "add_signature")      
builder.add_edge("add_signature", END)              

graph = builder.compile()                            

result = graph.invoke({"name": "Aiman", "greeting": "", "shout": "", "final": ""})  

print(result["final"])                              