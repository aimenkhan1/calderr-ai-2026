"""
 Document Processing Graph (LangGraph)
Load (PDF) -> Validate -> [conditional: Split if too large] -> Chunk -> Embed -> Confirm

"""


from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader


#state type for the graph
class State(TypedDict):
    doc_path: str           # input: path to the PDF file
    doc_text: str            # raw extracted text
    doc_size: int            # character count of doc_text
    is_valid: bool           # result of validation
    parts: List[str]         # split parts (only used if doc was too large)
    chunks: List[str]        # final chunks ready for embedding
    embeddings: List[list]   # fake embeddings (list of vectors)->just sample
    status: str              # final confirmation or err msg


#node functions for the graph

#it reads a PDF file from disk and extracts its text content. It returns a dictionary containing the extracted text and its size in characters.
def load(state: State) -> dict:
    reader = PdfReader(state["doc_path"])
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""   # some pages may return None
    return {
        "doc_text": text,
        "doc_size": len(text),
    }

#it validates the extracted text by checking if it is non-empty. It returns a dictionary indicating whether the document is valid or not.
def validate(state: State) -> dict:
    is_valid = state["doc_size"] > 0
    return {"is_valid": is_valid}

#it determines the next step in the processing pipeline based on the validation result and document size. If the document is invalid, it routes to "invalid". If the document is too large (greater than 10,000 characters), it routes to "too_large". Otherwise, it routes to "normal".
def route_after_validate(state: State) -> str:
    if not state["is_valid"]:
        return "invalid"
    if state["doc_size"] > 10000:  
        return "too_large"
    return "normal"

#it runs only if the document is too large. It splits the extracted text into smaller parts of 5000 characters each and returns a dictionary containing the list of parts.
def split(state: State) -> dict:
    text = state["doc_text"]
    size = 5000
    parts = [text[i:i + size] for i in range(0, len(text), size)]
    return {"parts": parts}

#it chunks either the split parts (if they exist) or the raw document text into smaller chunks of 500 characters each. It returns a dictionary containing the list of chunks.
def chunk(state: State) -> dict:
    source = state.get("parts") or [state["doc_text"]]
    chunks = []
    for part in source:
        chunks += [part[i:i + 500] for i in range(0, len(part), 500)]
    return {"chunks": chunks}

#it just creates a placeholder embedding for each chunk by returning the length of each chunk as a vector. It returns a dictionary containing the list of embeddings.
def embed(state: State) -> dict:
    embeddings = [[len(c)] for c in state["chunks"]]
    return {"embeddings": embeddings}


def confirm(state: State) -> dict:
    return {"status": f" Done — {len(state['chunks'])} chunks embedded."}


def invalid_doc(state: State) -> dict:
    return {"status": " Document invalid or unreadable — aborted."}


#build the graph
builder = StateGraph(State)

builder.add_node("load", load)
builder.add_node("validate", validate)
builder.add_node("split", split)
builder.add_node("chunk", chunk)
builder.add_node("embed", embed)
builder.add_node("confirm", confirm)
builder.add_node("invalid_doc", invalid_doc)

builder.add_edge(START, "load")
builder.add_edge("load", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "invalid": "invalid_doc",
        "too_large": "split",
        "normal": "chunk",
    },
)

builder.add_edge("split", "chunk")
builder.add_edge("chunk", "embed")
builder.add_edge("embed", "confirm")
builder.add_edge("confirm", END)
builder.add_edge("invalid_doc", END)

graph = builder.compile()       
png_bytes = graph.get_graph().draw_mermaid_png()    #this saves png in current folder or we can go to studio also thru langgraph dev
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("Graph diagram saved as graph.png")


#main
if __name__ == "__main__":
    doc_path = input("Enter the path to your PDF file: ").strip().strip('"').strip("'")
    result = graph.invoke({
        "doc_path": doc_path,   
        "doc_text": "",
        "doc_size": 0,
        "is_valid": False,
        "parts": [],
        "chunks": [],
        "embeddings": [],
        "status": "",
    })

    print(result["status"])
    print("Doc size (chars):", result["doc_size"])
    print("Was split into parts:", len(result["parts"]))
    print("Number of chunks:", len(result["chunks"]))