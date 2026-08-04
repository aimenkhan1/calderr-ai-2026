"""
Generates graph.png - a visual diagram of the main graph.
"""

from graph import build_graph

if __name__ == "__main__":
    graph = build_graph()
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph.png")