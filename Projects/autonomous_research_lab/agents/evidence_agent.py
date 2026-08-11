"""
Evidence Agent — instantiated dynamically per SpecialistSpec (3-5 times
per run, never a fixed class per domain). Each instance:
  1. Retrieves relevant chunks from the seed document store via TF-IDF (RAG)
  2. Optionally calls a `fetch_full_document` tool if the excerpt isn't enough
  3. Produces a structured, cited EvidenceFinding
"""

from llm_client import maybe_call_tool, structured_completion
from models import EvidenceFinding, SpecialistSpec, _EvidenceOutput
from rag.document_store import DocumentStore


class EvidenceAgent:

    def __init__(self, spec: SpecialistSpec, domain: str, store: DocumentStore, finding_id: str):
        self.spec = spec
        self.domain = domain
        self.store = store
        self.finding_id = finding_id

    def _persona_system_prompt(self) -> str:
        return (
            f"You are {self.spec.persona_name}, a research specialist. Your expertise: "
            f"{self.spec.expertise_description}. You investigate ONLY your assigned "
            "sub-question — stay in your lane. Ground every claim in the provided "
            "source excerpts; do not use outside knowledge not present in the sources. "
            "If the sources don't fully answer the sub-question, say so honestly in "
            "your summary and lower your confidence accordingly rather than filling "
            "gaps with speculation."
        )

    def investigate(self) -> EvidenceFinding:
        retrieved = self.store.retrieve(self.domain, self.spec.sub_question, top_k=2)
        excerpts_blob = "\n\n".join(
            f"[Source: {r.source}] (relevance={r.score:.2f})\n{r.text}" for r in retrieved
        )
        used_tool = False

        # Give the model a chance to request the full document if excerpts are thin
        tool_context = ""
        if retrieved:
            def tool_executor(filename: str):
                return self.store.fetch_full_document(self.domain, filename)

            tool_context = maybe_call_tool(
                system_prompt=self._persona_system_prompt(),
                user_prompt=(
                    f"Sub-question: {self.spec.sub_question}\n\n"
                    f"Retrieved excerpts:\n{excerpts_blob}\n\n"
                    "If these excerpts are sufficient, just say so. If you need the "
                    "full text of one of these sources to answer confidently, call "
                    "the fetch_full_document tool."
                ),
                tool_executor=tool_executor,
            )
            used_tool = bool(tool_context)

        full_context = excerpts_blob + (f"\n\n{tool_context}" if tool_context else "")

        user_prompt = (
            f"Sub-question: {self.spec.sub_question}\n\n"
            f"Available source material:\n{full_context if full_context else '(no relevant sources found)'}"
        )

        result: _EvidenceOutput = structured_completion(
            self._persona_system_prompt(), user_prompt, _EvidenceOutput
        )

        return EvidenceFinding(
            finding_id=self.finding_id,
            persona_name=self.spec.persona_name,
            sub_question=self.spec.sub_question,
            summary=result.summary,
            supporting_points=result.supporting_points,
            confidence=result.confidence,
            sources=result.sources or [r.source for r in retrieved],
            used_tool_call=used_tool,
        )
