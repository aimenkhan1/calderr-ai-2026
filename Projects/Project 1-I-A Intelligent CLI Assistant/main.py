"""
Project 1-I-A: Intelligent CLI Assistant


Flow:
Load .env → Create Groq Client → Show Welcome → Show Topic Menu →
User Chooses Topic → Create Memory → User Asks Question →
Save to History → Send to Groq → Get Response → Save Reply →
Display in Rich Panel → Repeat Until /switch or /exit

Concepts Covered:
- Groq Chat Completions
- System Prompts and Topic Domains
- Multi-turn Conversation Memory
- Domain Guard (refuses off-topic questions)
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

# Topic Domains
topics = {

    "1": {
        "name": "Programming",
        "emoji": "👨‍💻",
        "system": """You are an expert programming assistant.
You help with Python, AI development, web development, and algorithms.
Always provide code examples when relevant.
Keep explanations clear and beginner friendly.
If asked about anything outside programming, say:
I am a programming assistant. Please ask me programming questions only."""
    },

    "2": {
        "name": "Cooking",
        "emoji": "🍳",
        "system": """You are an expert chef and cooking assistant.
You help with recipes, ingredients, cooking techniques, and meal planning.
Always mention cooking time and difficulty level when giving recipes.
If asked about anything outside cooking, say:
I am a cooking assistant. Please ask me cooking questions only."""
    },

    "3": {
        "name": "History",
        "emoji": "📚",
        "system": """You are an expert history teacher.
You help with world history, historical events, civilizations, and famous figures.
Always mention dates and context when explaining events.
If asked about anything outside history, say:
I am a history assistant. Please ask me history questions only."""
    }

}


# Chat Function
def chat(topic):

    history = [
        {"role": "system", "content": topic["system"]}
    ]

    total_tokens = 0
    message_count = 0

    console.print(Panel(
        f"{topic['emoji']} {topic['name']} Assistant is ready!",
        title="Active Topic"
    ))

    console.print("\n[dim]Commands: /clear | /history | /switch | /exit[/dim]\n")

    while True:

        question = console.input("[cyan]You: [/cyan]").strip()

        if not question:
            continue

        if question == "/exit":
            console.print(f"\n[dim]Session: {message_count} messages | {total_tokens} tokens[/dim]\n")
            return "exit"

        if question == "/switch":
            console.print(f"\n[dim]Session: {message_count} messages | {total_tokens} tokens[/dim]\n")
            return "switch"

        if question == "/clear":
            history = [{"role": "system", "content": topic["system"]}]
            total_tokens = 0
            message_count = 0
            console.print("[green]Memory cleared![/green]\n")
            continue

        if question == "/history":
            if len(history) == 1:
                console.print("[dim]No messages yet![/dim]\n")
            else:
                console.print("\n[bold]Chat History:[/bold]")
                for msg in history[1:]:
                    if msg["role"] == "user":
                        console.print(f"[cyan]You:[/cyan] {msg['content']}")
                    else:
                        console.print(f"[white]{topic['emoji']}:[/white] {msg['content'][:150]}...")
                console.print()
            continue

        history.append({"role": "user", "content": question})

        try:

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens
            total_tokens += tokens
            message_count += 1

            history.append({"role": "assistant", "content": answer})

            console.print(Panel(
                answer,
                title=f"{topic['emoji']} {topic['name']} Assistant"
            ))

            console.print(f"[dim]Tokens: {tokens} | Total: {total_tokens} | Messages: {message_count}[/dim]\n")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            history.pop()


# Main Program
console.print(Panel(
    "Intelligent CLI Assistant\nGroq + Rich + Topic Domains + Memory",
    title="Week 1 Project"
))

while True:

    console.print("\n[bold]Choose a Topic:[/bold]")
    console.print("1. 👨‍💻 Programming")
    console.print("2. 🍳 Cooking")
    console.print("3. 📚 History")
    console.print("[dim]/exit to quit[/dim]")

    choice = console.input("\n[cyan]Choice: [/cyan]").strip()

    if choice == "/exit":
        console.print("\n[cyan]Goodbye! 👋[/cyan]")
        break

    if choice not in topics:
        console.print("[red]Invalid! Enter 1, 2 or 3[/red]")
        continue

    result = chat(topics[choice])

    if result == "exit":
        console.print("\n[cyan]Goodbye! 👋[/cyan]")
        break