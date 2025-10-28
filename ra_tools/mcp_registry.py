"""MCP Tool Registry for discovering and managing MCP server connections."""

from typing import Dict, List, Optional, Any
import subprocess
import json


class MCPRegistry:
    """Registry for discovering and managing MCP server connections."""

    def __init__(self):
        """Initialize MCP registry."""
        self.available_servers: Dict[str, Dict[str, Any]] = {}
        self.discover_mcp_servers()

    def discover_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Auto-discover available MCP servers.

        Returns:
            Dictionary of available MCP servers with their capabilities
        """
        # This is a stub implementation
        # In production, would scan MCP server configuration
        # and validate availability

        self.available_servers = {
            "figma": {
                "available": False,  # Would check actual availability
                "description": "Figma MCP Server for design context",
                "tools": ["figma_get_file", "figma_get_components"],
                "config_required": True,
            },
            "v0": {
                "available": False,
                "description": "Vercel v0 MCP Server for UI generation",
                "tools": ["v0_generate_ui", "v0_generate_from_image", "v0_chat_complete"],
                "config_required": True,
            },
            "sequential-thinking": {
                "available": True,  # Assume available
                "description": "Advanced reasoning MCP tool",
                "tools": ["sequentialthinking"],
                "config_required": False,
            },
            "playwright": {
                "available": False,
                "description": "Browser automation MCP tool",
                "tools": ["browser_navigate", "browser_click", "browser_snapshot"],
                "config_required": False,
            },
        }

        return self.available_servers

    def is_server_available(self, server_name: str) -> bool:
        """Check if an MCP server is available.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if server is available
        """
        if server_name not in self.available_servers:
            return False

        return self.available_servers[server_name]["available"]

    def get_server_tools(self, server_name: str) -> List[str]:
        """Get list of tools provided by an MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of tool names
        """
        if server_name not in self.available_servers:
            return []

        return self.available_servers[server_name].get("tools", [])

    def validate_tool_availability(self, tool_name: str) -> bool:
        """Validate if a specific tool is available.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is available
        """
        for server in self.available_servers.values():
            if tool_name in server.get("tools", []) and server.get("available", False):
                return True

        return False

    def get_configuration_requirements(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration requirements for an MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Configuration requirements or None
        """
        if server_name not in self.available_servers:
            return None

        server = self.available_servers[server_name]

        if not server.get("config_required", False):
            return None

        # Return configuration template
        config_templates = {
            "figma": {
                "required_env": ["FIGMA_ACCESS_TOKEN"],
                "optional_env": ["FIGMA_FILE_ID"],
                "setup_instructions": "Get access token from Figma Settings > Personal Access Tokens",
            },
            "v0": {
                "required_env": ["V0_API_KEY"],
                "setup_instructions": "Get API key from Vercel v0 dashboard",
            },
        }

        return config_templates.get(server_name)

    def get_fallback_options(self, tool_name: str) -> List[str]:
        """Get fallback options if a tool is unavailable.

        Args:
            tool_name: Name of the requested tool

        Returns:
            List of alternative approaches
        """
        fallbacks = {
            "figma_get_file": [
                "Create design specifications in markdown",
                "Use Mermaid diagrams for wireframes",
                "Document design manually with screenshots",
            ],
            "v0_generate_ui": [
                "Write component specifications",
                "Create HTML/CSS mockups",
                "Use alternative design-to-code tools (Builder.io, Anima)",
            ],
        }

        return fallbacks.get(tool_name, ["Manual implementation required"])
