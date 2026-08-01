"""
Generates graph.png - a visual diagram of the research engine's architecture.
"""

from main import graph

if __name__ == "__main__":
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph.png")