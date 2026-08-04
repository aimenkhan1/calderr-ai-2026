"""
News Research Agent

Build: a news research agent that searches news, extracts entities,
classifies topics, and summarizes findings - using multiple API tools
chained together.

This file builds on top of external_apis.py - it imports the
real get_news() tool from there rather than duplicating it, and adds
three new "processing" tools that work on the search results:

    1. get_news            (imported)  -> real NewsAPI.org search
    2. extract_entities                -> pulls out likely names/places/orgs
    3. classify_topic                  -> labels text with a topic category
    4. summarize_text                  -> shortens text to key sentences

Two ways to use this file:
    A) run_tests()             -> exercises the tools + the chat agent
    B) news_research_agent()    -> runs the FULL fixed pipeline:
       search news -> extract entities -> classify topic -> summarize
       for every article found, then compiles a report

"""

import os
import re
import time
import json
from collections import Counter

from dotenv import load_dotenv
from groq import Groq, BadRequestError

from external_apis import get_news  

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


"""
Tool - Entity Extraction (lightweight heuristic, no external NLP library)
Picks out likely proper nouns (capitalized word sequences) as candidate
people/organizations/places.
"""

COMMON_SENTENCE_STARTERS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "I", "In", "On", "At", "But", "And", "However", "According"
}

def extract_entities(text: str) -> dict:

    if not text or not text.strip():
        return {"error": "No text provided"}

    # Matches sequences of 1+ capitalized words (e.g. "New York", "Elon Musk")
    candidates = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)

    entities = [c for c in candidates if c not in COMMON_SENTENCE_STARTERS]

    counted = Counter(entities)
    ranked = [name for name, _ in counted.most_common(10)]

    return {"entities_found": len(ranked), "entities": ranked}


"""
Tool - Topic Classification (keyword-based)
"""

TOPIC_KEYWORDS = {
    "Technology": ["ai", "software", "tech", "app", "chip", "computer", "startup", "robot"],
    "Business":   ["market", "stock", "economy", "company", "trade", "revenue", "investor"],
    "Politics":   ["election", "government", "president", "senate", "policy", "minister"],
    "Sports":     ["match", "tournament", "league", "player", "coach", "championship"],
    "Health":     ["health", "hospital", "disease", "vaccine", "medicine", "doctor"],
    "Science":    ["research", "study", "scientist", "space", "nasa", "discovery"],
}

def classify_topic(text: str) -> dict:

    if not text or not text.strip():
        return {"error": "No text provided"}

    text_lower = text.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return {"topic": "General", "confidence": "low"}

    top_topic = max(scores, key=scores.get)
    return {"topic": top_topic, "matched_keywords": scores[top_topic]}


"""
Tool - Summarize Text 
"""

def summarize_text(text: str, max_sentences: int = 2) -> dict:

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"error": "No text provided"}

    if len(sentences) <= max_sentences:
        return {"summary": text}

    step = max(1, len(sentences) // max_sentences)
    picked = [sentences[i * step] for i in range(max_sentences)]

    return {"summary": " ".join(picked)}


"""
Tool Schemas (for the free-form chat agent below)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search recent real news articles about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to search news for"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract likely people, places, and organization names from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_topic",
            "description": "Classify text into a topic category (Technology, Business, Politics, Sports, Health, Science, or General).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a piece of text into fewer sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }
]


def run_tool(tool_name: str, tool_args: dict) -> dict:
    print(f"    Tool     : {tool_name}")
    print(f"    Arguments: {tool_args}")

    dispatch = {
        "get_news": get_news,
        "extract_entities": extract_entities,
        "classify_topic": classify_topic,
        "summarize_text": summarize_text,
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}

    return fn(**tool_args)


"""
Groq retry wrapper -  Groq's Llama models occasionally emit a malformed tool call and throw tool_use_failed.
Retry a couple times, then fall back to a plain-text answer.
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
                print(f"    [retry {attempt + 1}/{max_retries}] Malformed tool call, retrying...")
                time.sleep(0.5 * (attempt + 1))
                continue
            raise

    raise last_error


def agent(user_input: str) -> str:
    print()
    print("User:", user_input)
    print()

    messages = [{"role": "user", "content": user_input}]

    response = call_with_tool_retry(messages, TOOLS)
    ai_message = response.choices[0].message

    if not ai_message.tool_calls:
        print("Agent:", ai_message.content)
        return ai_message.content

    tool_names = [tc.function.name for tc in ai_message.tool_calls]
    print(f"Tools selected: {tool_names}")
    print()

    messages.append(ai_message)

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        tool_args = json.loads(raw_args) if raw_args else {}

        tool_result = run_tool(tool_name, tool_args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result)
        })

    print()

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="none"
    )
    final_answer = final_response.choices[0].message.content
    print("Agent:", final_answer)
    return final_answer


"""
News Research Agent - the full fixed pipeline

Search news -> for each article: extract entities, classify topic,
summarize -> compile a readable report.

This is deliberately NOT routed through the LLM tool-picker - it's a
fixed pipeline that ALWAYS runs all 4 steps in order for every article,
because a research agent for this task needs to reliably do the same
analysis every time, not leave it up to the models discretion.
"""

def news_research_agent(topic: str, max_articles: int = 5) -> dict:

    print()
    print(f"NEWS RESEARCH AGENT - Topic: '{topic}'")
    print()

    news_result = get_news(topic, max_results=max_articles)

    if "error" in news_result:
        print(f"Failed to fetch news: {news_result['error']}")
        return news_result

    if news_result["found"] == 0:
        print("No articles found for this topic.")
        return news_result

    report = {"topic": topic, "articles_analyzed": [], "topic_distribution": Counter()}

    for i, article in enumerate(news_result["articles"], start=1):
        text_to_analyze = f"{article['title']}. {article['description']}"

        print(f"\n[{i}] {article['title']}")
        print(f"Source: {article['source']} | Published: {article['published_at']}")

        entities_result = extract_entities(text_to_analyze)
        topic_result = classify_topic(text_to_analyze)
        summary_result = summarize_text(text_to_analyze, max_sentences=1)

        entities = entities_result.get("entities", [])
        detected_topic = topic_result.get("topic", "General")
        summary = summary_result.get("summary", "")

        print(f"Entities : {', '.join(entities) if entities else 'none detected'}")
        print(f"Category : {detected_topic}")
        print(f"Summary  : {summary}")

        report["articles_analyzed"].append({
            "title": article["title"],
            "source": article["source"],
            "url": article["url"],
            "entities": entities,
            "category": detected_topic,
            "summary": summary
        })
        report["topic_distribution"][detected_topic] += 1

    print()
    print()
    print(f"Analyzed {len(report['articles_analyzed'])} articles.")
    print(f"Category breakdown: {dict(report['topic_distribution'])}")

    report["topic_distribution"] = dict(report["topic_distribution"])
    return report


"""
Tests
"""

def run_tests():

    print()
    print("DAY 4 - NEWS RESEARCH AGENT")
    print("Testing entity/topic/summary tools + the full pipeline")
    print()

    print("PHASE 1 - Individual processing tools")
    agent("Extract entities from: Elon Musk announced that Tesla and SpaceX are expanding operations in Texas.")
    agent("Classify this: The stock market rallied today as investors reacted to new trade policy.")
    agent("Summarize this: Researchers at NASA discovered a new exoplanet. The planet orbits a distant star. Scientists believe it may support life. Further study is planned for next year.")


    print()
    print("PHASE 2 - Full News Research Agent pipeline")
    user_topic = input("Enter a topic to search news for: ")
    news_research_agent(user_topic, max_articles=3)


if __name__ == "__main__":
    run_tests()