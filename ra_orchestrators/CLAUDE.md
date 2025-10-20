# Repository Analyzer Framework - Claude Code Instructions

This file provides guidance to Claude Code when working with the **Repository Analyzer Framework** - a portable, drop-in analysis toolkit for comprehensive repository analysis across multiple domains.

## Framework Overview

The Repository Analyzer Framework is a **standalone, reusable system** designed to be dropped into any repository for deep analysis. It provides multi-domain orchestration with specialized agents for:

- **Architecture Analysis** - Code structure, patterns, diagrams, data flows, API documentation
- **UX/UI Design** - User research, information architecture, visual design, prototyping
- **DevOps** (Future) - Infrastructure analysis, CI/CD workflows, IaC generation
- **Testing** (Future) - Test strategy, coverage analysis, test generation

### Key Design Principles

1. **Portability** - Drop into any repository without modification
2. **No Collisions** - `ra_` prefix avoids conflicts with existing code
3. **Timestamped Outputs** - Each run creates `ra_output/{domain}_{YYYYMMDD_HHMMSS}/`
4. **Extensibility** - Base framework supports new domains in <1 day
5. **Reusability** - Agents and tools shared across domains

## Directory Structure

```
ra_orchestrators/          # Orchestrator implementations
├── base_orchestrator.py   # Core framework (343 lines)
├── architecture_orchestrator.py
├── ux_orchestrator.py
├── CLAUDE.md             # This file
└── README.md             # Usage documentation

ra_agents/                # Agent definitions (JSON)
├── architecture/
│   ├── analyzer.json
│   └── doc_writer.json
└── ux/
    ├── ux_researcher.json
    ├── ia_architect.json
    ├── ui_designer.json
    └── prototype_developer.json

ra_tools/                 # Tool integrations
├── mcp_registry.py       # MCP server discovery
└── figma_integration.py  # Figma MCP + REST API

ra_output/                # Analysis outputs (timestamped)
├── architecture_20251003_122700/
├── architecture_20251003_153200/
├── ux_20251003_140000/
└── ux_20251003_161500/
```

## Common Commands

### Running Orchestrators

```bash
# Architecture analysis (generates ra_output/architecture_{timestamp}/)
python -m ra_orchestrators.architecture_orchestrator "Project Name"

# UX design workflow (generates ra_output/ux_{timestamp}/)
python -m ra_orchestrators.ux_orchestrator "Project Name"

# With timeout for long-running analyses
timeout 1800 python -m ra_orchestrators.architecture_orchestrator "Project Name"
```

### Agent Management

```bash
# List available agents
python -c "
from ra_agents.registry import AgentRegistry
registry = AgentRegistry()
print(registry.discover_agents())
"

# List agents for specific domain
python -c "
from ra_agents.registry import AgentRegistry
registry = AgentRegistry()
print(registry.discover_agents(domain='ux'))
"
```

### Framework Installation in Target Repository

```bash
# Option 1: Git submodule (recommended for ongoing updates)
cd /path/to/target/repo
git submodule add https://github.com/org/repo-analyzer-framework
git submodule update --init --recursive

# Option 2: Direct clone (for one-time analysis)
cd /path/to/target/repo
git clone https://github.com/org/repo-analyzer-framework ra_orchestrators
git clone https://github.com/org/repo-analyzer-framework/ra_agents ra_agents
git clone https://github.com/org/repo-analyzer-framework/ra_tools ra_tools

# Option 3: Download release (no git dependency)
curl -L https://github.com/org/repo-analyzer-framework/releases/latest/download/framework.tar.gz | tar xz

# Add to target repo's .gitignore
cat ra_orchestrators/.gitignore-template >> .gitignore
```

## Architecture Patterns

### Base Orchestrator Pattern

All orchestrators inherit from `BaseOrchestrator`:

```python
from ra_orchestrators.base_orchestrator import BaseOrchestrator
from claude_agent_sdk import AgentDefinition

class CustomOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(
            domain_name="custom",
            output_base_dir=Path("ra_output"),
            use_timestamp=True
        )

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        return {"agent_name": AgentDefinition(...)}

    def get_allowed_tools(self) -> List[str]:
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def run(self):
        await self.execute_phase("phase_1", "agent_name", "Analyze...", self.client)
        await self.execute_phase("phase_2", "agent_name", "Document...", self.client)
```

### Agent Definition Pattern (JSON)

Create reusable agents in `ra_agents/{domain}/{agent_name}.json`:

```json
{
  "name": "agent_name",
  "description": "What this agent does",
  "prompt": "Detailed system prompt explaining the agent's role and responsibilities...",
  "tools": ["Read", "Write", "Grep", "Glob"],
  "model": "sonnet",
  "domain": "custom",
  "version": "1.0.0"
}
```

Then load in orchestrator:

```python
from ra_agents.registry import AgentRegistry

registry = AgentRegistry()
agent = registry.load_agent("agent_name", domain="custom")
```

### Timestamped Output Pattern

The framework automatically creates timestamped directories:

- **Format**: `ra_output/{domain}_{YYYYMMDD_HHMMSS}/`
- **Example**: `ra_output/architecture_20251003_122754/`
- **Benefit**: Multiple analyses don't overwrite each other

To disable timestamps (legacy compatibility):

```python
orchestrator = ArchitectureOrchestrator(use_timestamp=False)
# Creates: ra_output/architecture_analysis/
```

## Adding a New Domain Orchestrator

Target: Implement new domain in <1 day

### Step 1: Create Orchestrator Class

```python
# ra_orchestrators/custom_orchestrator.py
from pathlib import Path
from typing import Dict, List
from claude_agent_sdk import AgentDefinition
from .base_orchestrator import BaseOrchestrator

class CustomOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(domain_name="custom")

        # Define subdirectories
        self.phase1_dir = self.output_dir / "01_phase1"
        self.phase2_dir = self.output_dir / "02_phase2"

        # Create structure
        self.create_output_structure()

    def create_output_structure(self, subdirs=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase1_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir.mkdir(parents=True, exist_ok=True)

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        return {
            "analyzer": AgentDefinition(
                description="Analyzes X",
                prompt="You are an expert in...",
                tools=["Read", "Write", "Grep"],
                model="sonnet"
            ),
            "doc_writer": AgentDefinition(
                description="Documents findings",
                prompt="You document...",
                tools=["Read", "Write"],
                model="sonnet"
            )
        }

    def get_allowed_tools(self) -> List[str]:
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def run(self):
        self.display_phase_header(1, "Analysis Phase", "🔍")
        await self.execute_phase(
            "analysis", "analyzer",
            "Analyze the repository and write results to {self.phase1_dir}/",
            self.client
        )

        self.display_phase_header(2, "Documentation Phase", "📝")
        await self.execute_phase(
            "documentation", "doc_writer",
            "Document findings in {self.phase2_dir}/",
            self.client
        )

if __name__ == "__main__":
    import asyncio
    orchestrator = CustomOrchestrator()
    asyncio.run(orchestrator.run_with_client())
```

### Step 2: Create Agent Definitions

```bash
mkdir -p ra_agents/custom
```

Create `ra_agents/custom/analyzer.json`:
```json
{
  "name": "analyzer",
  "description": "Analyzes code patterns and structure",
  "prompt": "You are an expert code analyzer. Focus on patterns, anti-patterns, and architecture decisions.",
  "tools": ["Read", "Write", "Grep", "Glob"],
  "model": "sonnet",
  "domain": "custom"
}
```

### Step 3: Run and Test

```bash
python -m ra_orchestrators.custom_orchestrator

# Output: ra_output/custom_20251003_154530/
#   ├── 01_phase1/
#   ├── 02_phase2/
#   └── ...
```

## Tool Integration

### MCP Server Discovery

```python
from ra_tools.mcp_registry import MCPRegistry

registry = MCPRegistry()
available_servers = registry.discover_servers()
print(f"Found {len(available_servers)} MCP servers")
```

### Figma Integration

```bash
# Set environment variable
export FIGMA_ACCESS_TOKEN="your_token"

# Or configure Figma MCP in Claude Code
# See ra_tools/figma_integration.py for detailed setup
```

In orchestrator:

```python
from ra_tools.figma_integration import FigmaIntegration

figma = FigmaIntegration()
if figma.is_available():
    # Use Figma MCP or REST API
    design_data = await figma.get_file(file_key)
```

## Best Practices

### Agent Design
1. **Single Responsibility** - Each agent has one clear purpose
2. **Explicit Tools** - Only include tools the agent needs
3. **File Writing Mandate** - Agents MUST use Write tool, not describe output
4. **Clear Prompts** - Include examples and edge cases

### Orchestrator Design
1. **Phase Independence** - Phases should be self-contained
2. **Output Verification** - Always verify expected files were created
3. **Progress Visibility** - Show tool usage for transparency
4. **Error Recovery** - Handle failures gracefully, preserve partial results

### Output Quality
1. **Structured Markdown** - Consistent heading levels and formatting
2. **Mermaid Diagrams** - Visualize architecture and flows
3. **Source References** - Link to files with line numbers (`file.py:123`)
4. **Examples** - Include code examples and usage patterns

## Performance & Cost

**Benchmarks (approximate):**
- Architecture Analysis: 5-10 minutes, $1-3
- UX Design Workflow: 10-20 minutes, $3-8
- Costs vary with codebase size and complexity

**Optimization:**
- Use concurrent phase execution when dependencies allow
- Enable context compaction for large codebases
- Cache agent responses for repeated analyses
- Use targeted prompts to reduce token usage

## Cross-Orchestrator Communication (Future)

Orchestrators will be able to invoke each other for validation and integration:

```python
# UX orchestrator validates API contracts with Architecture orchestrator
ux_orchestrator.invoke_orchestrator(
    orchestrator_name="architecture",
    phase_name="validate_api_contracts",
    context={"api_spec": api_data}
)
```

**Use Cases:**
- UX → Architecture (API contract validation)
- UX → Testing (E2E test scenario generation)
- Architecture → DevOps (deployment feasibility analysis)
- All → Documentation (centralized knowledge base)

## Troubleshooting

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'ra_orchestrators'
# Solution: Ensure you're running from repository root
cd /path/to/repo/root
python -m ra_orchestrators.architecture_orchestrator
```

### Agent Not Writing Files

Ensure agent prompt includes:
```
IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.
```

### Timestamp Collision

If running multiple analyses rapidly:
```python
import time
orchestrator1 = ArchitectureOrchestrator()
time.sleep(2)  # Ensure different timestamp
orchestrator2 = ArchitectureOrchestrator()
```

### MCP Tools Not Available

Check Claude Code MCP server configuration:
- Figma MCP Server
- Sequential Thinking
- Playwright (for browser automation)

See `ra_tools/` for integration examples.

## Distribution Model

### Standalone Repository

The framework is designed to be distributed as a standalone repository:

```
repo-analyzer-framework/
├── README.md (installation and usage)
├── CLAUDE.md (this file)
├── ra_orchestrators/
├── ra_agents/
├── ra_tools/
└── .gitignore-template
```

### Usage in Target Repositories

1. **Clone/submodule into target repo**
2. **Add ra_* to target's .gitignore**
3. **Run orchestrators from target repo root**
4. **Commit analysis outputs if desired**

### Version Management

- Framework has independent versioning
- Agents have domain-specific versions
- Outputs include framework version in metadata

## Related Documentation

- `README.md` - User-facing usage guide
- `claude-agents-research.md` - Comprehensive research (832 lines)
- Parent repo's `CLAUDE.md` - Project-specific instructions

## Contributing

When extending the framework:

1. Follow `BaseOrchestrator` patterns
2. Add agent JSON definitions to `ra_agents/{domain}/`
3. Update `README.md` with usage examples
4. Add tests for new functionality
5. Update this CLAUDE.md with new patterns

## License

Same as parent project (lila-mcp)
