# 🤖 Intelligent CLI Assistant

A terminal-based AI assistant powered by Groq and LangChain that answers 
questions across three topic domains with full conversation memory and 
a clean Rich terminal interface.

---

## 📸 What It Does

Choose a topic domain and the assistant will:
- Answer questions staying strictly within that domain
- Remember full conversation history across 10+ turns
- Show token usage per message and session total
- Allow switching between topics anytime
- Refuse off-topic questions with a polite message
- Display clean formatted output using Rich panels

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Groq (llama-3.1-8b-instant) |
| AI Framework | LangChain LCEL |
| Memory | LangChain Conversation Buffer |
| Terminal UI | Rich |
| Environment | python-dotenv |
| Language | Python 3.11+ |

---

## 📂 Project Structure
week1-cli-assistant/

│

├── main.py           # Single file application

├── requirements.txt  # Dependencies

└── README.md         # This file

---

## 🎯 Topic Domains

| Domain | Emoji | Description |
|--------|-------|-------------|
| Programming | 👨‍💻 | Python, AI, web dev, algorithms |
| Cooking | 🍳 | Recipes, ingredients, techniques |
| History | 📚 | World history, events, civilizations |

---

## 🏗 Architecture
User Input

↓

Command Check (/clear /switch /history /exit)

↓

LangChain ChatPromptTemplate

↓

Conversation Buffer Memory (HumanMessage + AIMessage list)

↓

LangChain Chain (prompt | ChatGroq)

↓

Groq API (llama-3.1-8b-instant)

↓

Rich Panel Output + Token Display

↓

Memory Updated

↓

Wait for next input ↑

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/aimenkhan1/calderr-ai-2026.git
cd calderr-ai-2026/projects/week1-cli-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root:
GROQ_API_KEY=your_groq_api_key_here
Get your free key at console.groq.com

### 4. Run the assistant
```bash
python main.py
```

---

## ⌨️ Commands

| Command | Action |
|---------|--------|
| `/clear` | Reset conversation memory |
| `/switch` | Change topic domain |
| `/history` | View full chat history |
| `/exit` | Quit the application |

---

## 💬 Example Conversations

### 👨‍💻 Programming
You: What is a Python decorator?

Assistant: A decorator is a function that wraps another function

to extend its behavior without modifying it...
You: Can you show me a simple example?

Assistant: Sure! Here's a basic decorator... [remembers context]
You: How would I use that for logging?

Assistant: Great question! Building on the example above... [still remembers]

### 🍳 Cooking
You: How do I make pasta carbonara?

Assistant: Classic carbonara needs eggs, pecorino, guanciale...

Cooking time: 20 minutes | Difficulty: Medium
You: What if I don't have guanciale?

Assistant: You can substitute with pancetta or bacon... [remembers dish]
You: How much pasta for 4 people?

Assistant: For 4 people you need about 400g of pasta... [still in context]

### 📚 History
You: Tell me about the Roman Empire

Assistant: The Roman Empire was one of the largest empires in history,

at its peak covering 5 million km²...
You: When did it fall?

Assistant: The Western Roman Empire fell in 476 AD when... [remembers topic]
You: Who was the last emperor?

Assistant: The last Western Roman Emperor was Romulus Augustulus... [in context]

---

## 📝 Notes

- Conversation memory uses LangChain's HumanMessage and AIMessage objects
- The model itself is stateless — full history is sent on every API call
- Domain guard is built into each topic's system prompt
- Memory is cleared completely with /clear command
- Switching topics starts a fresh memory for the new topic

---

## 👩‍💻 Built By

Aiman Nadeem Khan
