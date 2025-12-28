"""
Beam Agent - Advanced Multi-Perspective Reasoning Engine.

This agent implements a "Beam" style reasoning process where a task is analyzed from
multiple generated perspectives (divergence) and then synthesized into a cohesive result (convergence).
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .base_agent import ArchonDependencies, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class BeamDependencies(ArchonDependencies):
    """Dependencies for beam operations."""
    beam_width: int = 3  # Number of perspectives to generate
    model: str = "openai:gpt-4o"  # Model to use for sub-agents


class Perspective(BaseModel):
    """A generated perspective for the beam."""
    name: str = Field(description="Name of the perspective (e.g., 'Security Expert')")
    description: str = Field(description="Description of the perspective's focus and persona")
    rationale: str = Field(description="Why this perspective is relevant to the task")


class BeamDivergence(BaseModel):
    """Output of the divergence phase."""
    perspectives: List[Perspective] = Field(description="List of diverse perspectives to adopt")


class BeamSynthesis(BaseModel):
    """Output of the synthesis phase."""
    synthesis_thought_process: str = Field(description="Reflection on the different perspectives and how to merge them")
    final_response: str = Field(description="The final, synthesized answer")
    winning_perspective: str | None = Field(description="The perspective that contributed most (optional)")


# Simple wrapper to mimic PydanticAI RunResult structure for compatibility
class BeamRunResult:
    def __init__(self, data: BeamSynthesis):
        self.data = data


class BeamAgent(BaseAgent[BeamDependencies, BeamSynthesis]):
    """
    Beam Agent that performs multi-perspective reasoning.

    Process:
    1. Diverge: Generate N diverse perspectives based on the prompt.
    2. Parallel Execution: Run N agents in parallel, one for each perspective.
    3. Converge: Synthesize the results into one superior answer.
    """

    def __init__(self, model: str = None, **kwargs):
        super().__init__(model=model, name="BeamAgent", **kwargs)

    def _create_agent(self, **kwargs) -> Agent:
        """
        The BeamAgent's main agent is actually the 'Orchestrator'.
        It doesn't use a single PydanticAI agent for the whole flow because
        the flow involves parallel execution of *other* agents.
        """
        # The main agent here is just a placeholder to satisfy the BaseAgent contract.
        # We override `run` to handle the complex flow.
        return Agent(
            model=self.model,
            deps_type=BeamDependencies,
            result_type=BeamSynthesis,
            system_prompt="You are the Beam Orchestrator.",
            **kwargs
        )

    async def run(
        self,
        user_prompt: str,
        deps: BeamDependencies,
        message_history: Optional[List[Any]] = None,
        **kwargs
    ) -> BeamRunResult:
        """
        Execute the Beam Process.

        Args:
            user_prompt: The user's input prompt.
            deps: Dependencies for the agent.
            message_history: History of messages (ignored for now as Beam is single-turn logic).
            **kwargs: Additional arguments.

        Returns:
            BeamRunResult: Compatibility wrapper containing the BeamSynthesis data.
        """
        logger.info(f"Starting Beam Process for: {user_prompt[:50]}...")

        # 1. Divergence Phase
        perspectives = await self._diverge(user_prompt, deps)

        # 2. Parallel Execution Phase
        results = await self._execute_parallel(user_prompt, perspectives, deps)

        # 3. Convergence Phase
        synthesis = await self._converge(user_prompt, results, deps)

        # Return result wrapped for compatibility with server.py which expects .data attribute
        return BeamRunResult(data=synthesis)

    def run_stream(self, user_prompt: str, deps: BeamDependencies, **kwargs):
        """
        Run the agent with streaming output.

        Note: True streaming for Beam process is complex because of the parallel stages.
        For now, we will fallback to the default base implementation which streams
        the *Orchestrator* agent (the placeholder).

        TODO: Implement a custom generator that yields progress updates for the beam process.
        """
        # Since server.py calls this directly, we must return a context manager or async generator
        # compatible with PydanticAI.
        # The default implementation returns self._agent.run_stream(...)
        # But self._agent is just a placeholder.
        # So streaming "BeamAgent" will just behave like a dummy agent.

        # Ideally, we should raise an error or handle it.
        # But raising error crashes the stream endpoint.
        # Let's rely on the base implementation for now, which will just return
        # whatever the placeholder agent says ("I am the Beam Orchestrator").
        # This is not ideal but prevents a crash.

        # Better: let's modify the placeholder agent prompt to explain what's happening.
        self._agent = Agent(
            model=self.model,
            deps_type=BeamDependencies,
            system_prompt="You are the Beam Agent. Currently, streaming is not supported for the full multi-perspective process. Please use the standard run mode (non-streaming) to get the full synthesized result.",
        )
        return super().run_stream(user_prompt, deps, **kwargs)

    async def _diverge(self, prompt: str, deps: BeamDependencies) -> List[Perspective]:
        """Generate diverse perspectives."""

        divergence_agent = Agent(
            model=deps.model,
            result_type=BeamDivergence,
            system_prompt=f"""You are the Archon Beam Brain Controller.
Your goal is to analyze a user request and brainstorm {deps.beam_width} distinct, diverse, and relevant 'Perspectives' or 'Personas' to answer it.

**Guidelines:**
- **Diversity:** Perspectives should approach the problem from different angles (e.g., Technical, Ethical, Creative, Strategic).
- **Relevance:** Each perspective must add value to the specific user request.
- **Safety:** ALWAYS include one 'Ethical Overseer' or 'Safety Reviewer' perspective if the topic involves any ambiguity, risk, or moral decision.
- **Roleplay:** Define the persona clearly so the sub-agent knows how to behave.

Example for "How to hack a wifi?":
1. Network Security Expert (Focus on defensive mechanisms)
2. Ethical Overseer (Focus on legality and white-hat principles)
3. System Administrator (Focus on monitoring and logging)
"""
        )

        try:
            result = await divergence_agent.run(
                f"Generate {deps.beam_width} perspectives for this request: {prompt}",
            )
            logger.info(f"Generated {len(result.data.perspectives)} perspectives: {[p.name for p in result.data.perspectives]}")
            return result.data.perspectives
        except Exception as e:
            logger.error(f"Divergence failed: {e}")
            # Fallback perspectives
            return [
                Perspective(name="Direct Answer", description="Provide a clear, direct answer.", rationale="Standard response."),
                Perspective(name="Critical Reviewer", description="Critique the answer for accuracy and safety.", rationale="Quality control."),
                Perspective(name="Creative Thinker", description="Think outside the box.", rationale="Innovation."),
            ]

    async def _execute_parallel(self, prompt: str, perspectives: List[Perspective], deps: BeamDependencies) -> List[dict]:
        """Run agents in parallel for each perspective."""

        async def run_perspective(p: Perspective):
            # Create a temporary agent for this perspective
            sys_prompt = f"""You are acting as the '{p.name}' perspective.
**Description:** {p.description}
**Rationale:** {p.rationale}

Your task is to answer the user's request strictly from this perspective.
Do not break character. Provide the best possible answer according to your specific focus.
"""
            agent = Agent(
                model=deps.model,
                system_prompt=sys_prompt
            )

            try:
                # We expect a string response
                result = await agent.run(prompt)
                return {
                    "perspective": p,
                    "response": result.data,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Perspective '{p.name}' failed: {e}")
                return {
                    "perspective": p,
                    "response": f"Error generating response: {str(e)}",
                    "success": False
                }

        # Run all in parallel
        tasks = [run_perspective(p) for p in perspectives]
        results = await asyncio.gather(*tasks)
        return results

    async def _converge(self, prompt: str, results: List[dict], deps: BeamDependencies) -> BeamSynthesis:
        """Synthesize the parallel results into one answer."""

        # Prepare the context for synthesis
        perspectives_text = ""
        for res in results:
            p_name = res["perspective"].name
            content = res["response"]
            perspectives_text += f"\n--- PERSPECTIVE: {p_name} ---\n{content}\n"

        convergence_agent = Agent(
            model=deps.model,
            result_type=BeamSynthesis,
            system_prompt="""You are the Archon Beam Synthesizer.
You have received inputs from multiple expert perspectives regarding a user request.
Your job is to synthesize these inputs into a single, supreme response.

**Instructions:**
1. **Analyze:** Read all perspective outputs. Identify the best insights, corrections, and unique points from each.
2. **Resolve:** If perspectives conflict (e.g., Security vs. Usability), resolve the conflict with a balanced, reasoned judgement.
3. **Safety Check:** Pay special attention to the 'Ethical Overseer' or similar perspectives. If they raise valid concerns, incorporate warnings or refusals as appropriate.
4. **Synthesize:** Write a cohesive, high-quality response that blends the strengths of all perspectives. Do not just list them; integrate them.
5. **Output:** Return the reflection on the process and the final answer.
"""
        )

        try:
            result = await convergence_agent.run(
                f"Original User Request: {prompt}\n\nPerspective Outputs:{perspectives_text}"
            )
            return result.data
        except Exception as e:
            logger.error(f"Convergence failed: {e}")
            return BeamSynthesis(
                synthesis_thought_process="Synthesis failed, returning best effort.",
                final_response=f"I tried to synthesize multiple perspectives but encountered an error. Here are the raw outputs:\n{perspectives_text}",
                winning_perspective="None"
            )

    def get_system_prompt(self) -> str:
        return "You are the Beam Agent."
