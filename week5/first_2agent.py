"""
Week 5 -  First 2-Agent System
  A "Research Agent" fake-searches a few sources for info on a topic,
  then hands its findings to a "Synthesis Agent" which turns them into
  a neat report.

The two agents never pass raw text to each other. Everything they send
is a Pydantic "form" (a class with fixed fields) - if the form is filled
in wrong, Python refuses to create it. 
"""


from __future__ import annotations
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ValidationError
import random
import time




class SourceType(str, Enum):

    NEWS = "news"
    ACADEMIC = "academic"
    MARKET_DATA = "market_data"
    SOCIAL = "social"


class SourceFinding(BaseModel):
    
    source_type: SourceType         
    source_name: str       
    claim: str              
    confidence: float = Field(ge=0.0, le=1.0)
 

    


class ResearchRequest(BaseModel):

    topic: str
    sources_to_query: List[SourceType]


class ResearchFindings(BaseModel):

    topic: str                      
    findings: List[SourceFinding]        
    sources_queried: int            
    sources_failed: int              
    research_agent_notes: str       


class ReportSection(BaseModel):

    heading: str
    body: str
    supporting_confidence: float = Field(ge=0.0, le=1.0)


class StructuredReport(BaseModel):

    topic: str
    executive_summary: str      
    sections: List[ReportSection]  
    overall_confidence: float = Field(ge=0.0, le=1.0)
    caveats: List[str]     


MOCK_SOURCE_DB = {
    SourceType.NEWS: [
        ("TechCrunch", "Company announced a 20% YoY revenue increase last quarter.", 0.8),
        ("Reuters", "Regulatory scrutiny is increasing in the EU market.", 0.75),
    ],
    SourceType.ACADEMIC: [
        ("arXiv preprint", "Benchmark results show a 12% improvement over baseline.", 0.7),
    ],
    SourceType.MARKET_DATA: [
        ("Bloomberg terminal (mock)", "Stock is up 8% in the last 30 days.", 0.9),
    ],
    SourceType.SOCIAL: [
        ("Aggregated sentiment feed", "Public sentiment is mixed, trending slightly negative.", 0.5),
    ],
}


def query_mock_source(source_type: SourceType, topic: str) -> SourceFinding | None:

    time.sleep(0.05) 

    if random.random() < 0.15:
        return None

    entries = MOCK_SOURCE_DB.get(source_type, [])
    if not entries:
        return None

    name, claim_template, confidence = random.choice(entries)

    return SourceFinding(
        source_type=source_type,
        source_name=name,
        claim=f"[{topic}] {claim_template}",
        confidence=confidence,
    )



class ResearchAgent:
    name = "ResearchAgent"

    def run(self, request: ResearchRequest) -> ResearchFindings:

        assert isinstance(request, ResearchRequest)

        findings: List[SourceFinding] = []
        failed = 0

        for source in request.sources_to_query:
            result = query_mock_source(source, request.topic)
            if result is None:
                failed += 1        
            else:
                findings.append(result)  

    
        notes = (
            f"Queried {len(request.sources_to_query)} sources, "
            f"{failed} failed. "
            f"{'All findings collected successfully.' if failed == 0 else 'Proceeding with partial data.'}"
        )
        return ResearchFindings(
            topic=request.topic,
            findings=findings,
            sources_queried=len(request.sources_to_query),
            sources_failed=failed,
            research_agent_notes=notes,
        )


class SynthesisAgent:
    name = "SynthesisAgent"

    def run(self, findings: ResearchFindings) -> StructuredReport:
        assert isinstance(findings, ResearchFindings)

        if not findings.findings:
            return StructuredReport(
                topic=findings.topic,
                executive_summary="No findings were available to synthesize.",
                sections=[],
                overall_confidence=0.0,
                caveats=["All source queries failed or returned no data."],
            )


        by_type: dict[SourceType, List[SourceFinding]] = {}
        for f in findings.findings:
            by_type.setdefault(f.source_type, []).append(f)

        sections = []
        for source_type, items in by_type.items():
            # average the confidence of all findings in this group
            avg_conf = sum(i.confidence for i in items) / len(items)
            # combine all the claim sentences into one paragraph
            body = " ".join(i.claim for i in items)

            sections.append(ReportSection(
                heading=source_type.value.replace("_", " ").title(),
                body=body,
                supporting_confidence=round(avg_conf, 2),
            ))

        overall_confidence = round(
            sum(s.supporting_confidence for s in sections) / len(sections), 2
        )

        # build a list of warnings to show the reader
        caveats = []
        if findings.sources_failed > 0:
            caveats.append(
                f"{findings.sources_failed} of {findings.sources_queried} "
                f"source queries failed and are not reflected in this report."
            )
        low_conf_sections = [s.heading for s in sections if s.supporting_confidence < 0.6]
        if low_conf_sections:
            caveats.append(f"Low-confidence sections: {', '.join(low_conf_sections)}")

        summary = (
            f"Synthesized {len(findings.findings)} findings across "
            f"{len(by_type)} source categories for '{findings.topic}'. "
            f"Overall confidence: {overall_confidence}."
        )

        return StructuredReport(
            topic=findings.topic,
            executive_summary=summary,
            sections=sections,
            overall_confidence=overall_confidence,
            caveats=caveats,
        )




def run_pipeline(topic: str) -> StructuredReport:
    # Step 1: build the task/request
    request = ResearchRequest(
        topic=topic,
        sources_to_query=[
            SourceType.NEWS,
            SourceType.ACADEMIC,
            SourceType.MARKET_DATA,
            SourceType.SOCIAL,
        ],
    )

    researcher = ResearchAgent()
    synthesizer = SynthesisAgent()

    print(f"[{researcher.name}] received ResearchRequest(topic={request.topic!r})")

    findings = researcher.run(request)
    print(f"[{researcher.name}] -> [{synthesizer.name}]  ResearchFindings "
          f"({len(findings.findings)} findings, {findings.sources_failed} failed)")

    report = synthesizer.run(findings)
    print(f"[{synthesizer.name}] produced StructuredReport "
          f"(confidence={report.overall_confidence})")

    return report


def demonstrate_validation_error():

    print("\n--- Validation check: malformed message ---")
    try:
        SourceFinding(
            source_type=SourceType.NEWS,
            source_name="BadSource",
            claim="This should fail validation",
            confidence=1.5,   # <-- this is invalid on purpose
        )
    except ValidationError as e:
        print("Correctly rejected malformed message before any agent received it:")
        print(e)



if __name__ == "__main__":
    report = run_pipeline("Acme Corp market position")

    print()
    print("FINAL STRUCTURED REPORT")
    print()
    print(report.model_dump_json(indent=2))  

    demonstrate_validation_error()