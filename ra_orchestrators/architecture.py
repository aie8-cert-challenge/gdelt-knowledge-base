#!/usr/bin/env python3
"""Production Repository Analyzer using Claude Agent SDK.

This script performs comprehensive repository analysis with:
- Incremental file outputs for each phase
- Full progress visibility (tool uses and results)
- Structured output directory
- Phase-based analysis with checkpoints
"""

import asyncio
from pathlib import Path

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# Output directory structure
OUTPUT_DIR = Path("repo_analysis")
DOCS_DIR = OUTPUT_DIR / "docs"
DIAGRAMS_DIR = OUTPUT_DIR / "diagrams"
REPORTS_DIR = OUTPUT_DIR / "reports"


def display_message(msg, show_tools=True):
    """Display message content with full visibility into tool usage."""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"\n🤖 Claude: {block.text}")
            elif isinstance(block, ToolUseBlock) and show_tools:
                print(f"\n🔧 Using tool: {block.name}")
                if block.name in ["Read", "Grep", "Glob"]:
                    # Show what files are being analyzed
                    if "file_path" in block.input:
                        print(f"   Reading: {block.input['file_path']}")
                    if "pattern" in block.input:
                        print(f"   Pattern: {block.input['pattern']}")
                elif block.name == "Write":
                    # Show what files are being created
                    if "file_path" in block.input:
                        print(f"   ✍️  Writing: {block.input['file_path']}")
                elif block.name == "Bash":
                    # Show commands being executed
                    if "command" in block.input:
                        print(f"   Command: {block.input['command']}")

    elif isinstance(msg, UserMessage):
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and show_tools:
                # Show results of tool operations
                content = str(block.content)[:200] if block.content else "None"
                print(f"   ✅ Result: {content}...")

    elif isinstance(msg, ResultMessage):
        print("\n" + "=" * 70)
        print("✅ Phase completed")
        if msg.total_cost_usd and msg.total_cost_usd > 0:
            print(f"💰 Cost: ${msg.total_cost_usd:.4f}")
        print("=" * 70)


async def phase_1_component_inventory(client: ClaudeSDKClient):
    """Phase 1: Generate component inventory."""
    print("\n" + "=" * 70)
    print("📋 PHASE 1: Component Inventory")
    print("=" * 70)

    await client.query(
        f"""Use the analyzer agent to create a comprehensive component inventory.

Analyze the codebase and document:
1. All Python modules and their purposes
2. Key classes and functions with descriptions
3. Public API surface vs internal implementation
4. Entry points and main interfaces

Write your analysis to: {DOCS_DIR}/01_component_inventory.md

Use this structure:
# Component Inventory

## Public API
[List public modules, classes, functions]

## Internal Implementation
[List internal modules and their roles]

## Entry Points
[Document main entry points]

Include file paths and line numbers for all references."""
    )

    async for msg in client.receive_response():
        display_message(msg)


async def phase_2_architecture_diagrams(client: ClaudeSDKClient):
    """Phase 2: Generate architecture diagrams."""
    print("\n" + "=" * 70)
    print("🏗️  PHASE 2: Architecture Diagrams")
    print("=" * 70)

    await client.query(
        f"""Use the analyzer agent to create architecture diagrams.

Generate Mermaid diagrams showing:
1. System architecture (layered view)
2. Component relationships
3. Class hierarchies
4. Module dependencies

Write your diagrams to: {DIAGRAMS_DIR}/02_architecture_diagrams.md

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

Include explanations for each diagram."""
    )

    async for msg in client.receive_response():
        display_message(msg)


async def phase_3_data_flows(client: ClaudeSDKClient):
    """Phase 3: Document data flows."""
    print("\n" + "=" * 70)
    print("🔄 PHASE 3: Data Flow Analysis")
    print("=" * 70)

    await client.query(
        f"""Use the analyzer agent to document data flows.

Create sequence diagrams showing:
1. Simple query flow
2. Interactive client session flow
3. Tool permission callback flow
4. MCP server communication flow
5. Message parsing and routing

Write your analysis to: {DOCS_DIR}/03_data_flows.md

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

[Continue for other flows...]"""
    )

    async for msg in client.receive_response():
        display_message(msg)


async def phase_4_api_documentation(client: ClaudeSDKClient):
    """Phase 4: Generate API documentation."""
    print("\n" + "=" * 70)
    print("📚 PHASE 4: API Documentation")
    print("=" * 70)

    await client.query(
        f"""Use the doc-writer agent to create comprehensive API documentation.

Document:
1. All public functions and classes
2. Parameters, return types, and examples
3. Usage patterns and best practices
4. Configuration options

Write your documentation to: {DOCS_DIR}/04_api_reference.md

Use clear examples and link to source files."""
    )

    async for msg in client.receive_response():
        display_message(msg)


async def phase_5_synthesis(client: ClaudeSDKClient):
    """Phase 5: Create final synthesis document."""
    print("\n" + "=" * 70)
    print("📖 PHASE 5: Final Synthesis")
    print("=" * 70)

    await client.query(
        f"""Use the doc-writer agent to create a comprehensive README.

Review all previously generated documents in:
- {DOCS_DIR}/01_component_inventory.md
- {DIAGRAMS_DIR}/02_architecture_diagrams.md
- {DOCS_DIR}/03_data_flows.md
- {DOCS_DIR}/04_api_reference.md

Create a synthesis document at: {OUTPUT_DIR}/README.md

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

Make it comprehensive but accessible."""
    )

    async for msg in client.receive_response():
        display_message(msg)


async def verify_outputs():
    """Verify all expected outputs were created."""
    print("\n" + "=" * 70)
    print("🔍 Verifying Outputs")
    print("=" * 70)

    expected_files = [
        DOCS_DIR / "01_component_inventory.md",
        DIAGRAMS_DIR / "02_architecture_diagrams.md",
        DOCS_DIR / "03_data_flows.md",
        DOCS_DIR / "04_api_reference.md",
        OUTPUT_DIR / "README.md",
    ]

    for file_path in expected_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")

    print("=" * 70)


async def main():
    """Run comprehensive repository analysis in phases."""
    print("\n" + "=" * 70)
    print("🚀 Claude Agent SDK Repository Analyzer")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"- Documentation: {DOCS_DIR}")
    print(f"- Diagrams: {DIAGRAMS_DIR}")
    print(f"- Reports: {REPORTS_DIR}")
    print("\nThis will generate comprehensive repository documentation")
    print("in incremental phases with progress visibility.")
    print("=" * 70)

    # Define specialized agents
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

    options = ClaudeAgentOptions(
        agents={"analyzer": analyzer_agent, "doc-writer": doc_writer_agent},
        allowed_tools=["Read", "Write", "Grep", "Glob", "Bash"],
        permission_mode="acceptEdits",  # Auto-approve file writes
        cwd=".",  # Current directory
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Run analysis in phases
            await phase_1_component_inventory(client)
            await phase_2_architecture_diagrams(client)
            await phase_3_data_flows(client)
            await phase_4_api_documentation(client)
            await phase_5_synthesis(client)

        # Verify all outputs
        await verify_outputs()

        print(
            f"\n✅ Analysis complete! View results in: {OUTPUT_DIR.absolute()}/README.md"
        )

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\nPartial results may be available in:", OUTPUT_DIR.absolute())
        raise


if __name__ == "__main__":
    asyncio.run(main())
