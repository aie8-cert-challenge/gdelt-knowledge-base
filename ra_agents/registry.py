"""Agent registry for discovering and loading agent definitions."""

import json
from pathlib import Path
from typing import Dict, Optional

from claude_agent_sdk import AgentDefinition


class AgentRegistry:
    """Registry for discovering and loading agent definitions from JSON files."""

    def __init__(self, agents_dir: Path = Path(__file__).parent):
        """Initialize agent registry.

        Args:
            agents_dir: Base directory containing agent definitions
        """
        self.agents_dir = agents_dir
        self._cache: Dict[str, AgentDefinition] = {}

    def discover_agents(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Discover all available agent definition files.

        Args:
            domain: Optional domain filter (e.g., 'ux', 'architecture')

        Returns:
            Dictionary mapping agent names to file paths
        """
        agents = {}

        search_dirs = [self.agents_dir / domain] if domain else list(self.agents_dir.glob("*/"))

        for agent_dir in search_dirs:
            if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
                continue

            for agent_file in agent_dir.glob("*.json"):
                agent_name = agent_file.stem
                agents[agent_name] = str(agent_file)

        return agents

    def load_agent(self, agent_name: str, domain: Optional[str] = None) -> Optional[AgentDefinition]:
        """Load an agent definition from JSON file.

        Args:
            agent_name: Name of the agent (without .json extension)
            domain: Optional domain to search in

        Returns:
            AgentDefinition object or None if not found
        """
        # Check cache first
        cache_key = f"{domain}/{agent_name}" if domain else agent_name
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find agent file
        agents = self.discover_agents(domain)
        if agent_name not in agents:
            return None

        # Load and parse JSON
        agent_file = Path(agents[agent_name])
        with open(agent_file, "r") as f:
            agent_data = json.load(f)

        # Create AgentDefinition
        agent = AgentDefinition(
            description=agent_data["description"],
            prompt=agent_data["prompt"],
            tools=agent_data.get("tools", []),
            model=agent_data.get("model", "sonnet"),
        )

        # Cache and return
        self._cache[cache_key] = agent
        return agent

    def load_domain_agents(self, domain: str) -> Dict[str, AgentDefinition]:
        """Load all agents for a specific domain.

        Args:
            domain: Domain name (e.g., 'ux', 'architecture')

        Returns:
            Dictionary mapping agent names to AgentDefinition objects
        """
        agents = {}
        discovered = self.discover_agents(domain)

        for agent_name in discovered.keys():
            agent = self.load_agent(agent_name, domain)
            if agent:
                agents[agent_name] = agent

        return agents
