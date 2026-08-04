"""
5 Complex Pydantic Models
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------
# Model 1 - Job Posting
# ---------------------------------------------------

class RemoteStatus(str, Enum):
    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"


class JobPosting(BaseModel):
    title: str
    company: str

    salary_min: int = Field(ge=0)
    salary_max: int = Field(ge=0)

    required_skills: List[str]

    location: str

    remote_status: RemoteStatus

    @field_validator("salary_max")
    @classmethod
    def check_salary(cls, value, info):

        if "salary_min" in info.data:
            if value < info.data["salary_min"]:
                raise ValueError("salary_max must be greater than salary_min")

        return value


# ---------------------------------------------------
# Model 2 - Support Ticket
# ---------------------------------------------------

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class SupportTicket(BaseModel):
    customer_name: str
    issue_summary: str = Field(max_length=200)
    priority: Priority
    category: str
    requires_callback: bool = False


# ---------------------------------------------------
# Model 3 - Review Analysis
# ---------------------------------------------------

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class ReviewAnalysis(BaseModel):
    product_name: str

    sentiment: Sentiment

    rating: int = Field(ge=1, le=5)

    key_points: List[str]

    would_recommend: bool


# ---------------------------------------------------
# Model 4 - Meeting Notes
# ---------------------------------------------------

class ActionItem(BaseModel):
    task: str
    assigned_to: str
    deadline: Optional[str] = None


class MeetingNotes(BaseModel):
    meeting_title: str
    date: str
    attendees: List[str]
    action_items: List[ActionItem]
    summary: str = Field(max_length=500)


# ---------------------------------------------------
# Model 5 - Resume Parser
# ---------------------------------------------------

class Education(BaseModel):
    degree: str
    institution: str
    year: int


class ResumeData(BaseModel):
    full_name: str = Field(alias="name")
    email: Optional[str] = None

    years_experience: int = Field(ge=0)

    skills: List[str]

    education: List[Education]

    model_config = {
        "populate_by_name": True
    }