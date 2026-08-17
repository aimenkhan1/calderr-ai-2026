"""
seed_data.py

Populates a fresh MemoryAgent with a realistic scripted scenario (a
developer working with an AI coding assistant over a few sessions), so the
Memory Inspector UI has something meaningful to show on first launch instead
of four empty panels.

Deliberately includes 13+ plain episodic observations so episodic
consolidation actually fires once (see episodic.py) the UI will show
both the active log AND a consolidated summary block.
"""

from agent import MemoryAgent


SCRIPTED_OBSERVATIONS = [
    # profile / semantic facts 
    "My name is Alex.",
    "I am a backend engineer.",
    "I work at Nimbus Robotics.",
    "I live in Austin.",
    "I prefer Python over JavaScript.",
    "My favorite editor is Neovim.",
    "My timezone is America/Chicago.",

    # knowledge graph relations 
    "Alex works on Project Falcon.",
    "Project Falcon depends on the Auth Service.",
    "The Auth Service uses PostgreSQL.",
    "The Auth Service was built by the Platform Team.",
    "Project Falcon works with the Billing Service.",
    "The Billing Service depends on the Auth Service.",

    # procedural corrections 
    "Always include type hints in Python code.",
    "When writing SQL, never use SELECT star.",
    "From now on, summarize logs in bullet points.",
    "When reviewing pull requests, always check for missing tests.",

    # plain episodic events (no fact/relation/correction to extract) 
    "Investigated a slow query in the Billing Service dashboard.",
    "Paired with a teammate on the new auth token refresh flow.",
    "Deployed Project Falcon v2.3 to staging.",
    "Fixed a flaky integration test in the CI pipeline.",
    "Reviewed a pull request for the Platform Team.",
    "Discussed Q3 roadmap priorities in the planning meeting.",
    "Debugged a memory leak in the Auth Service.",
    "Wrote documentation for the Billing Service API.",
    "Rotated API keys for the staging environment.",
]

# Importance hints: corrections and identity facts matter more than routine chores.
IMPORTANCE_OVERRIDES = {
    "My name is Alex.": 0.9,
    "I work at Nimbus Robotics.": 0.8,
    "Deployed Project Falcon v2.3 to staging.": 0.7,
    "Debugged a memory leak in the Auth Service.": 0.65,
}


def create_demo_agent() -> MemoryAgent:
    agent = MemoryAgent()
    for text in SCRIPTED_OBSERVATIONS:
        importance = IMPORTANCE_OVERRIDES.get(text, 0.4)
        agent.observe(text, importance=importance)
    return agent


if __name__ == "__main__":
    demo_agent = create_demo_agent()
    print("Episodic active:", len(demo_agent.episodic.episodes))
    print("Episodic blocks:", len(demo_agent.episodic.memory_blocks))
    print("Semantic facts:", len(demo_agent.semantic.get_profile()))
    print("Graph entities:", len(demo_agent.graph.entities()))
    print("Graph triples:", len(demo_agent.graph.get_triples()))
    print("Procedural corrections:", len(demo_agent.procedural.get_active()))
