# Multi-Domain Agent Orchestrators

Extensible framework for running multi-agent workflows across software development domains.

## Overview

This orchestrator system extends the proven patterns from `architecture.py` into a reusable framework supporting multiple domains:

- **Architecture Analysis** - Repository structure, diagrams, data flows, API documentation
- **UX/UI Design** - User research, information architecture, visual design, prototyping
- **DevOps** (Future) - Infrastructure audit, CI/CD, IaC generation, compliance
- **Testing** (Future) - Test strategy, test generation, quality gates

## Quick Start

### Architecture Analysis

```bash
python -m ra_orchestrators.architecture_orchestrator "Project Name"
```

Generates comprehensive repository analysis in `ra_output/`:
- Component inventory
- Architecture diagrams
- Data flow analysis
- API documentation
- Final synthesis

### UX Design Workflow

```bash
python -m orchestrators.ux_orchestrator "My Project Name"
```

Generates complete UX design in `outputs/ux_design/`:
1. User research and personas
2. Information architecture
3. Visual design specifications
4. Interactive prototypes
5. API contracts
6. Design system documentation

## Architecture

### Base Orchestrator

`base_orchestrator.py` provides common functionality:

- **Phase Execution Engine** - Sequential or concurrent phase execution
- **Agent Lifecycle Management** - Load, configure, and delegate to specialized agents
- **Progress Tracking** - Visual progress with tool usage details
- **Cost Monitoring** - Track costs per phase and total
- **Output Verification** - Validate expected outputs were created
- **Error Handling** - Graceful error recovery and reporting

### Domain-Specific Orchestrators

Each domain orchestrator inherits from `BaseOrchestrator` and implements:

```python
class CustomOrchestrator(BaseOrchestrator):
    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        # Define specialized agents for this domain
        pass

    def get_allowed_tools(self) -> List[str]:
        # Specify available tools
        pass

    async def run(self):
        # Implement domain-specific workflow phases
        pass
```

### Agent Library

Agents are defined as JSON files in `agents/{domain}/`:

```json
{
  "name": "agent_name",
  "description": "What this agent does",
  "prompt": "Detailed system prompt...",
  "tools": ["Read", "Write", "Grep", "Glob"],
  "model": "sonnet",
  "domain": "ux",
  "version": "1.0.0"
}
```

**Available Agents:**
- `architecture/analyzer` - Code analysis and architecture
- `architecture/doc_writer` - Technical documentation (shared)
- `ux/ux_researcher` - User research and personas
- `ux/ia_architect` - Information architecture
- `ux/ui_designer` - Visual design
- `ux/prototype_developer` - Interactive prototyping

### Tool Integration

`tools/` provides integration layer for external services:

- **MCP Registry** - Discover and manage MCP server connections
- **Figma Integration** - Figma MCP and REST API wrapper
- **Builder.io** (Future) - Design-to-code integration
- **Anima API** (Future) - AI-optimized design-to-code

## Usage Examples

### Custom Project Name

```bash
python -m orchestrators.ux_orchestrator "E-Commerce Platform"
```

### Programmatic Usage

```python
from ra_orchestrators.ux_orchestrator import UXOrchestrator
import asyncio

async def main():
    orchestrator = UXOrchestrator(
        project_name="My App",
        output_base_dir=Path("custom_output"),
        show_tool_details=True
    )
    await orchestrator.run_with_client()

asyncio.run(main())
```

### Cross-Orchestrator Communication

```python
# Future: Orchestrators can call each other
ux_orchestrator.invoke_orchestrator(
    orchestrator_name="architecture",
    phase_name="validate_api_contracts",
    context={"api_spec": api_spec_data}
)
```

## Output Structure

### Architecture Analysis

```
ra_output/
├── docs/
│   ├── 01_component_inventory.md
│   ├── 03_data_flows.md
│   └── 04_api_reference.md
├── diagrams/
│   └── 02_architecture_diagrams.md
└── README.md
```

### UX Design

```
outputs/ux_design/
├── 01_research/
│   └── user_research.md
├── 02_ia/
│   └── information_architecture.md
├── 03_design/
│   └── visual_design.md
├── 04_prototypes/
│   └── interactive_prototypes.md
├── 05_api_contracts/
│   └── api_specifications.md
└── 06_design_system/
    └── design_system.md
```

## Adding a New Orchestrator

1. **Create orchestrator class:**

```python
# orchestrators/custom_orchestrator.py
from .base_orchestrator import BaseOrchestrator

class CustomOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(domain_name="custom")

    def get_agent_definitions(self):
        return {"agent1": AgentDefinition(...)}

    def get_allowed_tools(self):
        return ["Read", "Write", "Grep"]

    async def run(self):
        await self.phase_1_custom_task()
        await self.phase_2_another_task()
```

2. **Define agents:**

```bash
mkdir agents/custom
# Create agent JSON files
```

3. **Run:**

```bash
python -m orchestrators.custom_orchestrator
```

## Configuration

### Tool Access

Orchestrators auto-detect available tools. Configure MCP servers in Claude Code settings for enhanced capabilities:

- **Figma MCP** - Design context and component access
- **v0 MCP** - UI generation from natural language
- **Sequential Thinking** - Advanced reasoning
- **Playwright** - Browser automation

### Environment Variables

```bash
# Figma integration
export FIGMA_ACCESS_TOKEN="your_token"

# v0 integration
export V0_API_KEY="your_key"

# Telemetry (optional)
export ENABLE_LOGFIRE_TELEMETRY=true
```

## Best Practices

### Agent Design

1. **Single Responsibility** - Each agent should have one clear purpose
2. **Explicit Tool Lists** - Only include tools the agent needs
3. **Clear Prompts** - Detailed system prompts with examples
4. **File Writing** - Agents must use Write tool, not just describe output

### Orchestrator Design

1. **Phase Independence** - Each phase should be self-contained when possible
2. **Verification** - Always verify expected outputs were created
3. **Progress Visibility** - Display progress for long-running phases
4. **Error Recovery** - Handle failures gracefully with partial results

### Output Quality

1. **Structured Markdown** - Use consistent heading levels and formatting
2. **Mermaid Diagrams** - Visualize architecture, flows, and relationships
3. **Source Links** - Reference files with line numbers (`file.py:123`)
4. **Examples** - Include code examples and usage patterns

## Performance

**Benchmarks (approximate):**
- Architecture Analysis: 5-10 minutes, $1-3
- UX Design Workflow: 10-20 minutes, $3-8
- Cost varies with codebase size and complexity

**Optimization:**
- Use concurrent phase execution when possible
- Enable context compaction for long runs
- Cache agent responses where appropriate

## Extending the System

### Add New Domain

See `claude-agents-research.md` for comprehensive guidance on:
- Multi-agent orchestration patterns
- Tool integration strategies
- Agent design best practices
- Cross-domain communication

Target: Add new domain orchestrator in <1 day

### Agent Reusability

Agents can be shared across domains:

```python
# Use doc_writer from architecture domain in UX domain
from ra_agents.registry import AgentRegistry

registry = AgentRegistry()
doc_writer = registry.load_agent("doc_writer", domain="architecture")
```

## Troubleshooting

### MCP Tools Not Available

Check MCP server configuration in Claude Code settings. See `tools/figma_integration.py` for setup instructions.

### Agent Not Writing Files

Ensure agent prompt includes:
```
IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.
```

### High Costs

- Review phase complexity - break into smaller phases
- Use targeted prompts - be specific about requirements
- Enable progress tracking to identify expensive operations

## Related Documentation

- `claude-agents-research.md` - Comprehensive research on agent patterns and tools
- `architecture.py` - Original architecture analyzer (legacy)
- `CLAUDE.md` - Project-level Claude Code instructions

## Contributing

When adding orchestrators or agents:

1. Follow existing patterns in `base_orchestrator.py`
2. Add JSON agent definitions to `agents/{domain}/`
3. Update this README with usage examples
4. Add tests for new functionality

## License

Same as parent project (lila-mcp)
