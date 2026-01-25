import os
from crewai import Agent, Task, Crew, Process

SYSTEM_STYLE = """You are a helpful product search assistant.
Use the provided candidate items as your only source of truth.
If the filter is too strict, say so and suggest changing max_price or query.
Return recommendations in Spanish, with bullet points and a short rationale.
"""

def build_crew():
    agent = Agent(
        role="Product Retrieval Assistant",
        goal="Recommend the best items given a user query and candidate items.",
        backstory=SYSTEM_STYLE,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "User query: {query}\n\n"
            "Context (candidate items):\n{context}\n\n"
            "Deliverables:\n"
            "1) Top 5 recommendations with reason\n"
            "2) If fewer than 5 items, explain why and what to tweak\n"
        ),
        expected_output="A Spanish answer with top picks and reasoning.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    return crew
