"""
models.py - Data Structures

This file defines the shape of data used across the project.
Pydantic models make sure the AI always returns correctly structured data.

Models:
- ResearchPlan      : what the planner agent returns
- SubtopicResearch  : what each research step returns
- ResearchReport    : the final complete report
"""

from pydantic import BaseModel
from typing import List


class ResearchPlan(BaseModel):

    main_topic: str
    subtopics: List[str]
    research_goal: str


class SubtopicResearch(BaseModel):

    subtopic: str
    findings: str
    confidence: float
    key_points: List[str]


class ResearchReport(BaseModel):

    title: str
    summary: str
    sections: List[SubtopicResearch]
    conclusion: str
    overall_confidence: float