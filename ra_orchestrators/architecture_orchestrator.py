#!/usr/bin/env python3
"""Architecture Analysis Orchestrator using Claude Agent SDK.

Refactored from architecture.py to use the base orchestrator framework.
Performs comprehensive repository analysis with:
- Incremental file outputs for each phase
- Full progress visibility (tool uses and results)
- Structured output directory
- Phase-based analysis with checkpoints
"""

from pathlib import Path
from typing import Dict, List

from claude_agent_sdk import AgentDefinition

from .base_orchestrator import BaseOrchestrator


class ArchitectureOrchestrator(BaseOrchestrator):
    """Orchestrator for comprehensive repository architecture analysis."""

    def __init__(
        self,
        output_base_dir: Path = Path("ra_output"),
        show_tool_details: bool = True,
        use_timestamp: bool = True,
    ):
        """Initialize architecture orchestrator.

        Args:
            output_base_dir: Base directory for analysis outputs (default: ra_output)
            show_tool_details: Whether to display detailed tool usage
            use_timestamp: Whether to append timestamp to output directory
        """
        super().__init__(
            domain_name="architecture",
            output_base_dir=output_base_dir,
            show_tool_details=show_tool_details,
            use_timestamp=use_timestamp,
        )

        # Set up subdirectory structure
        # self.output_dir is already set by BaseOrchestrator as ra_output/architecture_{timestamp}/
        self.docs_dir = self.output_dir / "docs"
        self.diagrams_dir = self.output_dir / "diagrams"
        self.reports_dir = self.output_dir / "reports"

        # Create directory structure
        self.create_output_structure()

    def create_output_structure(self, subdirs: List[str] = None):
        """Create output directory structure for architecture analysis."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.diagrams_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        """Get agent definitions for architecture analysis.

        Returns:
            Dictionary of agent definitions
        """
        analyzer_agent = AgentDefinition(
            description="Analyzes code structure, patterns, and architecture",
            prompt="""You are a code analyzer expert. Your job is to:

1. Examine code structure, patterns, and architecture systematically
2. Generate clear Mermaid diagrams for visualization
3. Write comprehensive documentation with examples
4. Reference specific files and line numbers
5. Create well-structured markdown documents

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

Be thorough but concise. Focus on clarity and accuracy.""",
            tools=["Read", "Grep", "Glob", "Write", "Bash"],
            model="sonnet",
        )

        doc_writer_agent = AgentDefinition(
            description="Writes comprehensive technical documentation",
            prompt="""You are a technical documentation expert. Your job is to:

1. Write clear, comprehensive documentation with examples
2. Create well-organized markdown documents
3. Include diagrams where helpful
4. Focus on developer experience and clarity
5. Link to source files with specific line numbers

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

Make documentation accessible and practical.""",
            tools=["Read", "Write", "Grep", "Glob"],
            model="sonnet",
        )

        return {
            "analyzer": analyzer_agent,
            "doc-writer": doc_writer_agent,
        }

    def get_allowed_tools(self) -> List[str]:
        """Get list of allowed tools for architecture analysis.

        Returns:
            List of tool names
        """
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def phase_1_component_inventory(self):
        """Phase 1: Generate component inventory."""
        self.display_phase_header(1, "Component Inventory", "📋")

        await self.execute_phase(
            phase_name="Component Inventory",
            agent_name="analyzer",
            prompt=f"""Use the analyzer agent to create a comprehensive component inventory.

IMPORTANT: Only analyze the main project code.
EXCLUDE these framework directories from analysis:
- ra_orchestrators/ (analysis framework)
- ra_agents/ (analysis framework)
- ra_tools/ (analysis framework)
- ra_output/ (generated outputs)
- .venv/ (virtual environment)

Analyze the codebase and document:
1. All Python modules and their purposes (EXCLUDING ra_* directories)
2. Key classes and functions with descriptions
3. Public API surface vs internal implementation
4. Entry points and main interfaces

Write your analysis to: {self.docs_dir}/01_component_inventory.md

Use this structure:
# Component Inventory

## Public API
[List public modules, classes, functions]

## Internal Implementation
[List internal modules and their roles]

## Entry Points
[Document main entry points]

Include file paths and line numbers for all references.""",
            client=self.client,
        )

    async def phase_2_architecture_diagrams(self):
        """Phase 2: Generate architecture diagrams."""
        self.display_phase_header(2, "Architecture Diagrams", "🏗️")

        await self.execute_phase(
            phase_name="Architecture Diagrams",
            agent_name="analyzer",
            prompt=f"""Use the analyzer agent to create architecture diagrams.

IMPORTANT: Only diagram the main project code.
EXCLUDE these framework directories:
- ra_orchestrators/
- ra_agents/
- ra_tools/
- ra_output/
- .venv/

Generate Mermaid diagrams showing:
1. System architecture (layered view - main project only)
2. Component relationships (excluding ra_* framework)
3. Class hierarchies (project classes only)
4. Module dependencies (project modules only)

Write your diagrams to: {self.diagrams_dir}/02_architecture_diagrams.md

Use this structure:
# Architecture Diagrams

## System Architecture
```mermaid
[System architecture diagram]
```

## Component Relationships
```mermaid
[Component relationship diagram]
```

## Class Hierarchies
```mermaid
[Class hierarchy diagrams]
```

Include explanations for each diagram.""",
            client=self.client,
        )

    async def phase_3_data_flows(self):
        """Phase 3: Document data flows."""
        self.display_phase_header(3, "Data Flow Analysis", "🔄")

        await self.execute_phase(
            phase_name="Data Flow Analysis",
            agent_name="analyzer",
            prompt=f"""Use the analyzer agent to document data flows.

Create sequence diagrams showing:
1. Simple query flow
2. Interactive client session flow
3. Tool permission callback flow
4. MCP server communication flow
5. Message parsing and routing

Write your analysis to: {self.docs_dir}/03_data_flows.md

Use this structure:
# Data Flow Analysis

## Query Flow
```mermaid
[Sequence diagram]
```
[Explanation]

## Interactive Session Flow
```mermaid
[Sequence diagram]
```
[Explanation]

[Continue for other flows...]""",
            client=self.client,
        )

    async def phase_4_api_documentation(self):
        """Phase 4: Generate API documentation."""
        self.display_phase_header(4, "API Documentation", "📚")

        await self.execute_phase(
            phase_name="API Documentation",
            agent_name="doc-writer",
            prompt=f"""Use the doc-writer agent to create comprehensive API documentation.

Document:
1. All public functions and classes
2. Parameters, return types, and examples
3. Usage patterns and best practices
4. Configuration options

Write your documentation to: {self.docs_dir}/04_api_reference.md

Use clear examples and link to source files.""",
            client=self.client,
        )

    async def phase_5_synthesis(self):
        """Phase 5: Create final synthesis document."""
        self.display_phase_header(5, "Final Synthesis", "📖")

        await self.execute_phase(
            phase_name="Final Synthesis",
            agent_name="doc-writer",
            prompt=f"""Use the doc-writer agent to create a comprehensive README.

Review all previously generated documents in:
- {self.docs_dir}/01_component_inventory.md
- {self.diagrams_dir}/02_architecture_diagrams.md
- {self.docs_dir}/03_data_flows.md
- {self.docs_dir}/04_api_reference.md

Create a synthesis document at: {self.output_dir}/README.md

Structure:
# Repository Architecture Documentation

## Overview
[High-level summary]

## Quick Start
[How to use this documentation]

## Architecture Summary
[Key insights from diagrams]

## Component Overview
[Summary of components]

## Data Flows
[Key flow patterns]

## References
[Links to detailed docs]

Make it comprehensive but accessible.""",
            client=self.client,
        )

    async def run(self):
        """Run comprehensive repository analysis in phases."""
        # Run all analysis phases
        await self.phase_1_component_inventory()
        await self.phase_2_architecture_diagrams()
        await self.phase_3_data_flows()
        await self.phase_4_api_documentation()
        await self.phase_5_synthesis()

        # Verify all outputs
        expected_files = [
            self.docs_dir / "01_component_inventory.md",
            self.diagrams_dir / "02_architecture_diagrams.md",
            self.docs_dir / "03_data_flows.md",
            self.docs_dir / "04_api_reference.md",
            self.output_dir / "README.md",
        ]

        await self.verify_outputs(expected_files)


async def main():
    """Run comprehensive repository analysis."""
    orchestrator = ArchitectureOrchestrator()
    await orchestrator.run_with_client()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
