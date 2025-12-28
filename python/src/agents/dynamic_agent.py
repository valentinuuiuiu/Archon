"""
Dynamic Agent - A generic agent class for loading agents from the database.
"""

from typing import Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .base_agent import ArchonDependencies, BaseAgent

@dataclass
class DynamicDependencies(ArchonDependencies):
    """Dependencies for dynamic agents."""
    context: dict[str, Any] | None = None

class DynamicAgentResult(BaseModel):
    """Generic result for dynamic agents."""
    response: str
    data: dict[str, Any] | None = None

class DynamicAgent(BaseAgent[DynamicDependencies, DynamicAgentResult]):
    """
    An agent loaded dynamically from configuration.
    """

    def __init__(self, name: str, system_prompt: str, model: str, tools: list[str] = None, **kwargs):
        self._system_prompt_content = system_prompt
        self._tools_config = tools or []
        super().__init__(model=model, name=name, **kwargs)

    def _create_agent(self, **kwargs) -> Agent:
        agent = Agent(
            model=self.model,
            deps_type=DynamicDependencies,
            result_type=DynamicAgentResult,
            system_prompt=self._system_prompt_content,
            **kwargs
        )

        # Here we would register tools based on self._tools_config
        # For now, we don't have a registry of tools to map strings to functions.
        # This is a placeholder for future expansion.

        return agent

    def get_system_prompt(self) -> str:
        return self._system_prompt_content
