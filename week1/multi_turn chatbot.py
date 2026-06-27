"""
Start Program → Load .env → Create Groq Client → Create Rich Console → Show Welcome Screen → Show Persona Menu → User Chooses Persona → Create Memory (History) → Show Commands → User Asks Question → Save Question to History → Send Entire History to Groq → Groq Generates Response → Extract Answer & Tokens → Save AI Reply to History → Display Response in Rich Panel → Wait for Next Question → Repeat Until /switch or /exit

Week 1 Integration Project

This project combines all the concepts learned during Week 1 into a single chatbot.
It uses the Groq API for AI responses, supports multiple personas through system prompts,
maintains conversation memory for multi-turn chats, and provides a clean terminal
interface using the Rich library.

Concepts Covered:
- Groq Chat Completions
- System Prompts
- Multi-turn Conversation Memory
- Persona-based AI Assistants
- Rich Terminal UI

"""

import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
console = Console()

# Personas
personas = {
    "1": {
        "name": "AI Engineering Assistant",
        "emoji": "🤖",
        "system": """You help users learn AI, LangChain, Groq and Python.
Give simple explanations.
Provide practical examples.
Keep answers short."""
    },
    "2": {
        "name": "Python Code Reviewer",
        "emoji": "👨‍💻",
        "system": """You review Python code.
Find mistakes.
Suggest improvements.
Explain why."""
    },
    "3": {
        "name": "Data Analyst",
        "emoji": "📊",
        "system": """You explain data clearly.
Use bullet points.
End every answer with Key Insight:"""
    }
}

# Chat Function
def chat(persona):

    history = [
        {"role": "system", "content": persona["system"]}
    ]

    console.print(Panel(
        f"{persona['emoji']} {persona['name']} is ready!",
        title="Active Persona"
    ))

    console.print("\n[dim]Commands: /clear | /history | /switch | /exit[/dim]\n")

    while True:

        question = console.input("[cyan]You: [/cyan]").strip()

        if not question:
            continue

        if question == "/exit":
            return "exit"

        if question == "/switch":
            return "switch"

        if question == "/clear":
            history = [{"role": "system", "content": persona["system"]}]
            console.print("[green]Memory cleared![/green]")
            continue

        if question == "/history":
            console.print("\n[bold]Chat History:[/bold]")
            for msg in history[1:]:
                if msg["role"] == "user":
                    console.print(f"[cyan]You:[/cyan] {msg['content']}")
                else:
                    console.print(f"[white]{persona['emoji']}:[/white] {msg['content']}")
            console.print()
            continue

        history.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=history,
            temperature=0.7
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        history.append({"role": "assistant", "content": answer})

        console.print(Panel(
            answer,
            title=f"{persona['emoji']} {persona['name']}"
        ))

        console.print(f"[dim]Tokens used: {tokens}[/dim]\n")


# Main Program
console.print(Panel(
    "Week 1 Integration Project\nGroq + Rich + Personas + Memory",
    title="AI Chatbot"
))

while True:

    console.print("\n[bold]Choose a Persona:[/bold]")
    console.print("1. 🤖 AI Engineering Assistant")
    console.print("2. 👨‍💻 Python Code Reviewer")
    console.print("3. 📊 Data Analyst")
    console.print("[dim]/exit to quit[/dim]")

    choice = console.input("\n[cyan]Choice: [/cyan]").strip()

    if choice == "/exit":
        console.print("\n[cyan]Goodbye! 👋[/cyan]")
        break

    if choice not in personas:
        console.print("[red]Invalid choice! Enter 1, 2 or 3[/red]")
        continue

    result = chat(personas[choice])

    if result == "exit":
        console.print("\n[cyan]Goodbye! 👋[/cyan]")
        break
