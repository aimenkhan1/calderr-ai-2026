"""
Lab 2.2 - Multi Tool Research Agent

This program builds an AI agent that can use 8 different tools.
The AI reads the user's message and decides which tool to call.
Your Python code runs the tool and sends the result back to the AI.
The AI then gives a final answer in plain English.

Available Tools:
    1. search_db           - Search a mock employee database
    2. calculate           - Do math calculations
    3. format_date         - Parse and reformat dates
    4. convert_currency    - Convert between currencies
    5. summarize_text      - Shorten long text
    6. web_search_mock     - Search the web (mock)
    7. get_current_date    - Get today's date and time
    8. classify_sentiment  - Check if text is positive or negative

"""

import os
import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq, BadRequestError


#loading API key from .env file

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  =  "llama-3.3-70b-versatile"


""" 
Mock Data
These are the fake databases and indexes the tools use.
In a real project these would be real databases and APIs.
""" 

EMPLOYEE_DATABASE = [
    {"id": 1, "name": "Ali Hassan",     "role": "Engineer",  "salary": 85000},
    {"id": 2, "name": "Sara Khan",      "role": "Designer",  "salary": 72000},
    {"id": 3, "name": "Ahmed Raza",     "role": "Manager",   "salary": 95000},
    {"id": 4, "name": "Fatima Malik",   "role": "Engineer",  "salary": 88000},
    {"id": 5, "name": "Usman Sheikh",   "role": "Analyst",   "salary": 67000},
    {"id": 6, "name": "Zara Ahmed",     "role": "Designer",  "salary": 74000},
    {"id": 7, "name": "Bilal Chaudhry", "role": "Manager",   "salary": 98000},
    {"id": 8, "name": "Nadia Hussain",  "role": "Analyst",   "salary": 69000},
]

EXCHANGE_RATES = {
    "USD": 1.0,
    "PKR": 278.5,
    "EUR": 0.92,
    "GBP": 0.79,
    "AED": 3.67,
    "SAR": 3.75,
    "INR": 83.12,
    "CAD": 1.36,
    "JPY": 149.50,
}

WEB_INDEX = {
    "python":      "Python is a high-level programming language widely used in AI and data science.",
    "ai":          "Artificial Intelligence refers to machines that simulate human thinking.",
    "groq":        "Groq builds LPU hardware for ultra-fast AI inference.",
    "langchain":   "LangChain is a framework for building LLM-powered applications.",
    "pakistan":    "Pakistan is a country in South Asia with over 230 million people.",
    "llm":         "Large Language Models are AI models trained on massive text to understand language.",
    "pydantic":    "Pydantic is a Python library for data validation using type annotations.",
    "transformer": "The Transformer is the neural network architecture that powers modern LLMs.",
}

POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "happy", "love",
    "best", "wonderful", "fantastic", "brilliant", "awesome",
    "perfect", "beautiful", "glad", "superb", "outstanding"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "hate", "worst", "horrible",
    "poor", "disappointing", "sad", "angry", "broken",
    "failed", "useless", "wrong", "disgusting", "frustrating"
]


#Tool 1 - Search Employee Database -> it searches a mock employee database by name or job role and returns matching records. It handles case-insensitive queries and returns a not-found message if no matches exist.

def search_db(query: str) -> dict:

    query_lower = query.lower().strip()

    matches = [
        employee for employee in EMPLOYEE_DATABASE
        if query_lower in employee["name"].lower()
        or query_lower in employee["role"].lower()
    ]

    if matches:
        return {"found": len(matches), "records": matches}

    return {"found": 0, "message": f"No employees found matching '{query}'"}


#Tool 2 - Calculate Math Expression-it evaluates a math expression and returns the result. It handles basic arithmetic and parentheses. It also checks for invalid characters and division by zero.

def calculate(expression: str) -> dict:

    allowed = set("0123456789+-*/()., ")

    if not all(char in allowed for char in expression):
        return {"error": "Expression contains invalid characters"}

    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"error": "Cannot divide by zero"}
    except Exception as e:
        return {"error": f"Could not calculate: {str(e)}"}


#Tool 3(Date Formatter)->it parses a date string and returns it in a readable format along with the day of the week, month, and year. It tries multiple common date formats automatically.

def format_date(date: str, output_format: str = "%d %B %Y") -> dict:

    formats_to_try = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%d %B %Y",
    ]

    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(date.strip(), fmt)
            return {
                "original":  date,
                "formatted": parsed.strftime(output_format),
                "day":       parsed.strftime("%A"),
                "month":     parsed.strftime("%B"),
                "year":      parsed.year
            }
        except ValueError:
            continue

    return {"error": f"Could not understand the date '{date}'. Try formats like 2026-07-04"}


#Tool 4(Currency Converter)->it converts a money amount from one currency to another using fixed exchange rates based on USD

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:

    from_curr = from_currency.upper().strip()
    to_curr   = to_currency.upper().strip()

    if from_curr not in EXCHANGE_RATES:
        return {"error": f"Currency '{from_currency}' is not supported"}

    if to_curr not in EXCHANGE_RATES:
        return {"error": f"Currency '{to_currency}' is not supported"}

    # it converts source to USD first, then USD to target
    in_usd    = amount / EXCHANGE_RATES[from_curr]
    converted = in_usd * EXCHANGE_RATES[to_curr]
    rate      = EXCHANGE_RATES[to_curr] / EXCHANGE_RATES[from_curr]

    return {
        "amount":    amount,
        "from":      from_curr,
        "to":        to_curr,
        "converted": round(converted, 2),
        "rate":      round(rate, 4)
    }


#Tool 5(Summarize Text)->it shortens a long piece of text by picking key sentences and returns a shorter version along with compression stats

def summarize_text(text: str, max_sentences: int = 3) -> dict:

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"error": "No text provided"}

    if len(sentences) <= max_sentences:
        return {
            "summary":             text,
            "sentences_in_summary": len(sentences)
        }

    # It picks evenly spaced sentences across the text
    step    = max(1, len(sentences) // max_sentences)
    picked  = [sentences[i * step] for i in range(max_sentences)]
    summary = " ".join(picked)

    return {
        "summary":              summary,
        "sentences_in_summary": max_sentences,
        "original_sentences":   len(sentences),
        "compression_ratio":    round(len(summary) / len(text), 2)
    }


#Tool 6(Web Search Mock)->it searches a small mock web index for information about a topic and returns a result if the query matches a known keyword.

def web_search_mock(query: str) -> dict:
    query_lower = query.lower().strip()

    for keyword, result in WEB_INDEX.items():
        if keyword in query_lower:
            return {"query": query, "result": result}

    return {"query": query, "result": "No results found in the mock index."}


#Tool 7(Get Current Date)->it returns today's date, day of the week, time, and year

def get_current_date() -> dict:

    now = datetime.now()
    return {
        "date":      now.strftime("%Y-%m-%d"),
        "day":       now.strftime("%A"),
        "time":      now.strftime("%H:%M:%S"),
        "month":     now.strftime("%B"),
        "year":      now.year,
        "formatted": now.strftime("%d %B %Y")
    }


#Tool 8(Sentiment Classifier)->it classify the sentiment of the text as positive, negative, or neutral based on keyword matching


def classify_sentiment(text: str) -> dict:

    text_lower = text.lower()

    positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    total          = positive_count + negative_count

    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.5, "text": text}

    if positive_count > negative_count:
        sentiment  = "positive"
        confidence = round(positive_count / total, 2)
    elif negative_count > positive_count:
        sentiment  = "negative"
        confidence = round(negative_count / total, 2)
    else:
        sentiment  = "neutral"
        confidence = 0.5

    return {
        "text":       text[:80] + "..." if len(text) > 80 else text,
        "sentiment":  sentiment,
        "confidence": confidence
    }


""" 
Tool Schemas
This is what the AI reads to understand what tools exist.
The description field is the most important part.
The AI uses descriptions to decide which tool to call.
""" 

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_db",
            "description": "Search the employee database by name or job role. Use when user asks about employees or staff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name or role to search e.g. Engineer or Ali"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Use for any arithmetic or calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression e.g. 25 * 4"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_date",
            "description": "Parse and reformat a date string. Use when user gives a date and wants it formatted or wants the day of the week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date string e.g. 2026-07-04 or 15/08/2025"
                    }
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert money from one currency to another. Supports USD, PKR, EUR, GBP, AED, SAR, INR, CAD, JPY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount of money to convert"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code e.g. USD"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code e.g. PKR"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a long piece of text into fewer sentences. Use when user gives a paragraph and wants a shorter version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The full text to summarize"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_mock",
            "description": "Search the web for information about a topic. Use when user asks what something is or wants to learn about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for e.g. what is LangChain"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date, day of the week, and time. Use when user asks what today's date or day is.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_sentiment",
            "description": "Check if a piece of text is positive, negative, or neutral. Use when user wants to analyze tone or feeling of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

""" 
Tool Runner
Matches the tool name the AI selected to the
actual Python function and runs it.
"""

def run_tool(tool_name: str, tool_args: dict) -> dict:

    print(f"    Tool     : {tool_name}")
    print(f"    Arguments: {tool_args}")

    if tool_name == "search_db":
        return search_db(**tool_args)

    elif tool_name == "calculate":
        return calculate(**tool_args)

    elif tool_name == "format_date":
        return format_date(**tool_args)

    elif tool_name == "convert_currency":
        return convert_currency(**tool_args)

    elif tool_name == "summarize_text":
        return summarize_text(**tool_args)

    elif tool_name == "web_search_mock":
        return web_search_mock(**tool_args)

    elif tool_name == "get_current_date":
        return get_current_date()

    elif tool_name == "classify_sentiment":
        return classify_sentiment(**tool_args)

    else:
        return {"error": f"Unknown tool: {tool_name}"}


"""
Retry Wrapper
Groq's Llama models occasionally emit a malformed function-call string
instead of valid JSON, which raises a 400 'tool_use_failed' error.
This wrapper retries a couple of times, and on the last attempt forces
tool_choice='none' so the agent degrades to a plain-text answer instead
of crashing.
"""

def call_with_tool_retry(messages, tools, max_retries: int = 2):

    last_error = None

    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="none" if is_last_attempt else "auto"
            )
        except BadRequestError as e:
            last_error = e
            error_code = ""
            if hasattr(e, "body") and isinstance(e.body, dict):
                error_code = e.body.get("error", {}).get("code", "")

            if error_code == "tool_use_failed":
                print(f"    [retry {attempt + 1}/{max_retries}] Model produced a malformed tool call, retrying...")
                time.sleep(0.5 * (attempt + 1))  # simple backoff
                continue
            else:
                raise  # some other 400 error, don't swallow it

    # If we get here, even tool_choice='none' failed - genuinely unrecoverable
    raise last_error


"""
Agent
This is the main loop.
Step 1: Send user message to Groq with the tools list
Step 2: Check if AI wants to call a tool 
Step 3: Run the tool and collect the result
Step 4: Send the result back to Groq
Step 5: Get and print the final answer
"""

def agent(user_input: str) -> str:
    print()
    print("User:", user_input)
    print()

    messages = [{"role": "user", "content": user_input}]

    # Step 1: Ask Groq - do you need a tool?
    response = call_with_tool_retry(messages, TOOLS)

    ai_message = response.choices[0].message

    # Step 2: AI answered directly - no tool needed
    if not ai_message.tool_calls:
        print("Agent:", ai_message.content)
        return ai_message.content

    # Step 3: AI wants to use tools - show which ones
    tool_names = [tc.function.name for tc in ai_message.tool_calls]
    print(f"Tools selected: {tool_names}")
    print()

    # It adds AI's tool request to message history
    messages.append(ai_message)

    # Step 4: Run each tool the AI selected
    for tool_call in ai_message.tool_calls:

        tool_name = tool_call.function.name
        raw_args  = tool_call.function.arguments
        tool_args = json.loads(raw_args) if raw_args else {}

        tool_result = run_tool(tool_name, tool_args)

        # Add the tool result to message history
        messages.append({
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      json.dumps(tool_result)
        })

    print()

    # Step 5: Send results back to Groq for the final answer
    final_response = call_with_tool_retry(messages, TOOLS)

    final_answer = final_response.choices[0].message.content
    print("Agent:", final_answer)
    return final_answer

"""
Tests
Phase 1: Test each tool on its own
Phase 2: Test two or more tools at the same time
Phase 3: Verify the AI picks the correct tool
Phase 4: Verify the AI answers without a tool when not needed

"""

def run_tests():

    print()
    print("LAB 2.2 - MULTI TOOL RESEARCH AGENT")
    print("Testing all 8 tools across 4 phases")
    print()


    # Phase 1 - One tool at a time
    print("PHASE 1 - Each Tool On Its Own")
    print()

    agent("Find all Engineers in the database")

    agent("What is (450 * 12) + 350?")

    agent("What day of the week was 2026-07-04?")

    agent("Convert 500 USD to PKR")

    agent(
        "Summarize this: "
        "Artificial intelligence is transforming industries worldwide. "
        "Companies are investing billions in AI research. "
        "The technology automates tasks and improves decisions. "
        "However ethical concerns remain about job displacement. "
        "Experts urge careful regulation."
    )

    agent("What is Groq?")

    agent("What is today's date?")

    agent("Is this positive or negative: This product is absolutely amazing and wonderful!")


    # Phase 2 - Multiple tools at the same time
    print()
    print("PHASE 2 - Multiple Tools Together")
    print()

    agent("What is today's date AND convert 100 USD to EUR?")

    agent("Find all Managers AND calculate 95000 * 12")

    agent(
        "What is AI AND classify this: "
        "AI is a terrible threat to humanity"
    )


    # Phase 3 - Routing check
    print()
    print("PHASE 3 - Routing Check")
    print()

    agent("What day is it today?")
    agent("What is 999 divided by 3?")
    agent("Show me all Analysts")
    agent("Is this text positive or negative: I hate this broken software")
    agent("What is LangChain?")
    agent("Format this date: 15/08/2025")
    agent("Convert 1000 GBP to JPY")


    # Phase 4 - No tool needed
    print()
    print("PHASE 4 - No Tool Needed")
    print()

    agent("Hello how are you?")
    agent("What is the capital of France?")
    agent("Tell me a fun fact about space")


if __name__ == "__main__":
    run_tests()