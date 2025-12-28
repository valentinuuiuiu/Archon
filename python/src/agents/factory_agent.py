"""
Factory Agent - The "Agent Factory" capable of creating and managing other agents.

This agent uses the `archon_agents` table to persist agent definitions.
"""

import logging
import uuid
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .base_agent import ArchonDependencies, BaseAgent
from ..server.services.client_manager import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class FactoryDependencies(ArchonDependencies):
    """Dependencies for factory operations."""
    pass


class FactoryOperation(BaseModel):
    """Structured output for factory operations."""
    operation_type: str = Field(description="Type of operation: create_agent, delete_agent, list_agents")
    agent_name: str | None = Field(description="Name of the agent affected")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Human-readable message about the operation")
    agent_details: dict[str, Any] | None = Field(description="Details of the created/modified agent")


class FactoryAgent(BaseAgent[FactoryDependencies, FactoryOperation]):
    """
    Agent Factory capable of creating, updating, and deleting other agents.
    """

    def _create_agent(self, **kwargs) -> Agent:
        agent = Agent(
            model=self.model,
            deps_type=FactoryDependencies,
            result_type=FactoryOperation,
            system_prompt="""You are the Archon Agent Factory. Your purpose is to create, configure, and manage other AI agents.

**Your Capabilities:**
- **Create Agent:** Design and register a new agent with a specific role, system prompt, and tools.
- **List Agents:** Show all custom agents currently registered in the system.
- **Delete Agent:** Remove an agent from the system.

**How to Create an Agent:**
When a user asks to create an agent (e.g., "Create a Python Code Reviewer"), you should:
1.  **Analyze the Request:** Determine the agent's name, role, description, and necessary tools.
2.  **Draft a System Prompt:** Create a comprehensive system prompt that defines the agent's persona, constraints, and instructions.
3.  **Call `create_agent_tool`:** Use the tool to save the agent definition to the database.

**Agent Roles:**
- Use specific roles like "Code Reviewer", "QA Tester", "Documentation Writer", etc.

**Tools:**
- Currently, agents can be assigned a list of tools. For now, assume tools are just strings representing capabilities (e.g., "read_file", "search_web"). In the future, this will be more structured.
""",
            **kwargs,
        )

        @agent.tool
        async def create_agent_tool(
            ctx: RunContext[FactoryDependencies],
            name: str,
            role: str,
            description: str,
            system_prompt: str,
            model: str = "openai:gpt-4o",
            tools: list[str] = None,
        ) -> str:
            """
            Create and persist a new agent definition.

            Args:
                name: Unique name for the agent (e.g., "python-reviewer").
                role: The role/title of the agent.
                description: Short description of what the agent does.
                system_prompt: The full system prompt for the agent.
                model: The LLM model to use (default: openai:gpt-4o).
                tools: List of tool names this agent should have access to.
            """
            try:
                supabase = get_supabase_client()

                # Check if agent exists
                existing = supabase.table("archon_agents").select("id").eq("name", name).execute()
                if existing.data:
                    return f"Error: Agent with name '{name}' already exists."

                agent_data = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "role": role,
                    "description": description,
                    "system_prompt": system_prompt,
                    "model": model,
                    "tools": tools or [],
                    "created_by": ctx.deps.user_id or "FactoryAgent",
                }

                supabase.table("archon_agents").insert(agent_data).execute()
                return f"Successfully created agent '{name}'."

            except Exception as e:
                logger.error(f"Error creating agent: {e}")
                return f"Error creating agent: {str(e)}"

        @agent.tool
        async def list_agents_tool(ctx: RunContext[FactoryDependencies]) -> str:
            """List all custom agents."""
            try:
                supabase = get_supabase_client()
                response = supabase.table("archon_agents").select("name, role, description").execute()

                if not response.data:
                    return "No custom agents found."

                agents_list = []
                for a in response.data:
                    agents_list.append(f"- **{a['name']}** ({a['role']}): {a['description']}")

                return "\n".join(agents_list)
            except Exception as e:
                logger.error(f"Error listing agents: {e}")
                return f"Error listing agents: {str(e)}"

        @agent.tool
        async def delete_agent_tool(ctx: RunContext[FactoryDependencies], name: str) -> str:
            """Delete an agent by name."""
            try:
                supabase = get_supabase_client()
                response = supabase.table("archon_agents").delete().eq("name", name).execute()
                # Supabase delete returns the deleted rows
                if not response.data:
                    return f"Agent '{name}' not found."
                return f"Successfully deleted agent '{name}'."
            except Exception as e:
                logger.error(f"Error deleting agent: {e}")
                return f"Error deleting agent: {str(e)}"

        return agent

    def get_system_prompt(self) -> str:
        return """You are the Archon Agent Factory. Your purpose is to create, configure, and manage other AI agents."""
