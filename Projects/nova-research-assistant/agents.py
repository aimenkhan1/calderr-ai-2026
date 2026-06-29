"""
agent.py - Agent Logic

This file contains all the AI agent logic for the research assistant.

Flow:
Question → Planner Agent → Research Loop → Synthesis Agent → Report

Agents:
- planner_agent()   : breaks question into subtopics
- research_agent()  : researches one subtopic
- synthesis_agent() : combines all findings into a report
- run_research()    : runs all agents in order
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from models import ResearchPlan, SubtopicResearch, ResearchReport

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Helper function to call Groq

def call_groq(system, user, temperature=0.7):

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],

        temperature=temperature
    )

    return response.choices[0].message.content


# Clean JSON response from Groq

def clean_json(text):

    text = text.strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return text


# Step 1 - Planner Agent
def planner_agent(question):

    # Check if question is valid
    check_system = """You are a validator.
Decide if the input can be researched as a topic.

Reply with ONLY one word: VALID or INVALID

INVALID only if input is:
- Just a greeting: hello, hi, hey, thanks, ok, bye
- Random single words with no meaning: asdf, xyz
- Nonsense text

VALID if input is:
- Any question about any topic
- Any subject that can be explained or researched
- Even short topics like "langchain", "python", "AI"
- Anything that has information to look up

Examples:
hello → INVALID
what is langchain → VALID
langchain flow → VALID
machine learning → VALID
hi there → INVALID
what caused ww1 → VALID
vaccines → VALID
asdfgh → INVALID"""

    check = call_groq(check_system, question, temperature=0)

    if "INVALID" in check.upper() and "VALID" not in check.upper().replace("INVALID", ""):
        raise ValueError(
            "I'm a research assistant! 🤖 Please ask me a research question like:\n"
            "• What is machine learning?\n"
            "• How does LangChain work?\n"
            "• What caused World War 1?"
        )

    system = """You are a research planning agent.
Break the research question into 3 subtopics.

Reply in this EXACT JSON format with no extra text:
{
    "main_topic": "the main topic",
    "subtopics": ["subtopic 1", "subtopic 2", "subtopic 3"],
    "research_goal": "what we want to find out"
}

Only reply with the JSON. No extra text before or after."""

    response = call_groq(system, question, temperature=0)

    cleaned = clean_json(response)

    if not cleaned or cleaned.strip() == "":
        raise ValueError("Please enter a proper research question!")

    data = json.loads(cleaned)

    return ResearchPlan(
        main_topic=data["main_topic"],
        subtopics=data["subtopics"],
        research_goal=data["research_goal"]
    )


# Step 2 - Research Agent

def research_agent(subtopic, main_topic):

    system = """You are an expert research agent.

Research the given subtopic thoroughly.

Assign a meaningful confidence score based on how reliable and certain the information is.

Confidence Guidelines:
- 0.90–1.00 → Well-established facts with strong evidence.
- 0.70–0.89 → Mostly reliable information with minor uncertainty.
- 0.50–0.69 → Mixed evidence or some uncertainty.
- Below 0.50 → Speculative, uncertain, or limited information.

Reply in this EXACT JSON format with no extra text:
{
    "findings": "detailed findings in 2-3 sentences",
    "confidence": 0.85,
    "key_points": ["point 1", "point 2", "point 3"]
}

confidence must be a number between 0.0 and 1.0.
Only choose a high confidence score when the information is well-established."""

    user = f"Main topic: {main_topic}\nSubtopic: {subtopic}"

    response = call_groq(system, user, temperature=0.7)

    data = json.loads(clean_json(response))

    return SubtopicResearch(
        subtopic=subtopic,
        findings=data["findings"],
        confidence=float(data["confidence"]),
        key_points=data["key_points"]
    )


# Step 3 - Synthesis Agent

def synthesis_agent(plan, research_results):

    research_text = ""

    for result in research_results:
        research_text += f"\nSubtopic: {result.subtopic}\n"
        research_text += f"Findings: {result.findings}\n"
        research_text += f"Key Points: {', '.join(result.key_points)}\n"

    system = """You are a research synthesis agent.
Combine all findings into a clear report.

Reply in this EXACT JSON format with no extra text:
{
    "summary": "2-3 sentence overview of all findings",
    "conclusion": "final conclusion and key takeaways"
}"""

    user = f"""Topic: {plan.main_topic}
Goal: {plan.research_goal}

Research Findings:
{research_text}"""

    response = call_groq(system, user, temperature=0.7)

    data = json.loads(clean_json(response))

    average_confidence = sum(
        r.confidence for r in research_results
    ) / len(research_results)

    return ResearchReport(
        title=f"Research Report: {plan.main_topic}",
        summary=data["summary"],
        sections=research_results,
        conclusion=data["conclusion"],
        overall_confidence=round(average_confidence, 2)
    )


# Main Research Runner

def run_research(question, progress_callback=None):

    # Step 1 - Plan
    if progress_callback:
        progress_callback("Planning research strategy...", 0.1)

    plan = planner_agent(question)

    # Step 2 - Research each subtopic
    research_results = []
    total = len(plan.subtopics)

    for i, subtopic in enumerate(plan.subtopics):

        if progress_callback:
            progress = 0.2 + (0.6 * (i / total))
            progress_callback(f"Researching: {subtopic}", progress)

        result = research_agent(subtopic, plan.main_topic)
        research_results.append(result)

    # Step 3 - Synthesize
    if progress_callback:
        progress_callback("Synthesizing findings...", 0.9)

    report = synthesis_agent(plan, research_results)

    if progress_callback:
        progress_callback("Report complete!", 1.0)

    return report, plan