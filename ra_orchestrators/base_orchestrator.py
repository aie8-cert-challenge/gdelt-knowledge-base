#!/usr/bin/env python3
"""Base orchestrator framework for multi-domain agent workflows.

This module provides the foundational framework for all domain-specific orchestrators,
extracting common patterns from architecture.py into a reusable base class.

This framework is designed to be portable and can be dropped into any repository
for analysis purposes. Use the 'ra_' prefix to avoid naming collisions.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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


class BaseOrchestrator(ABC):
    """Base class for domain-specific orchestrators.

    Provides common functionality for:
    - Phase execution and management
    - Agent lifecycle management
    - Output directory structure
    - Progress tracking and visualization
    - Cost tracking and reporting
    - Verification and checkpointing
    - Error handling and recovery
    """

    def __init__(
        self,
        domain_name: str,
        output_base_dir: Path = Path("ra_output"),
        show_tool_details: bool = True,
        use_timestamp: bool = True,
    ):
        """Initialize base orchestrator.

        Args:
            domain_name: Name of the domain (e.g., 'architecture', 'ux', 'devops')
            output_base_dir: Base directory for all outputs (default: ra_output)
            show_tool_details: Whether to display detailed tool usage
            use_timestamp: Whether to append timestamp to output directory
        """
        self.domain_name = domain_name
        self.output_base_dir = output_base_dir

        # Create timestamped output directory: ra_output/{domain}_{YYYYMMDD_HHMMSS}/
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = output_base_dir / f"{domain_name}_{timestamp}"
        else:
            self.output_dir = output_base_dir / f"{domain_name}_analysis"

        self.show_tool_details = show_tool_details

        # Tracking
        self.total_cost = 0.0
        self.phase_costs: Dict[str, float] = {}
        self.completed_phases: List[str] = []

    def create_output_structure(self, subdirs: Optional[List[str]] = None):
        """Create output directory structure.

        Args:
            subdirs: Optional list of subdirectory names to create
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if subdirs:
            for subdir in subdirs:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def display_message(self, msg, show_tools: bool = True):
        """Display message content with full visibility into tool usage.

        Args:
            msg: Message to display (AssistantMessage, UserMessage, or ResultMessage)
            show_tools: Whether to show tool usage details
        """
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"\n🤖 Agent: {block.text}")
                elif isinstance(block, ToolUseBlock) and show_tools and self.show_tool_details:
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
                if isinstance(block, ToolResultBlock) and show_tools and self.show_tool_details:
                    # Show results of tool operations
                    content = str(block.content)[:200] if block.content else "None"
                    print(f"   ✅ Result: {content}...")

        elif isinstance(msg, ResultMessage):
            print("\n" + "=" * 70)
            print("✅ Phase completed")
            if msg.total_cost_usd and msg.total_cost_usd > 0:
                self.total_cost += msg.total_cost_usd
                print(f"💰 Cost: ${msg.total_cost_usd:.4f}")
            print("=" * 70)

    def display_phase_header(self, phase_number: int, phase_name: str, emoji: str = "📋"):
        """Display a formatted phase header.

        Args:
            phase_number: Phase number (1-indexed)
            phase_name: Name of the phase
            emoji: Emoji to display
        """
        print("\n" + "=" * 70)
        print(f"{emoji} PHASE {phase_number}: {phase_name}")
        print("=" * 70)

    def track_phase_cost(self, phase_name: str, cost: float):
        """Track cost for a specific phase.

        Args:
            phase_name: Name of the phase
            cost: Cost in USD
        """
        self.phase_costs[phase_name] = cost
        self.total_cost += cost

    def mark_phase_complete(self, phase_name: str):
        """Mark a phase as completed.

        Args:
            phase_name: Name of the completed phase
        """
        self.completed_phases.append(phase_name)

    async def verify_outputs(self, expected_files: List[Path]) -> bool:
        """Verify all expected outputs were created.

        Args:
            expected_files: List of expected file paths

        Returns:
            True if all files exist, False otherwise
        """
        print("\n" + "=" * 70)
        print("🔍 Verifying Outputs")
        print("=" * 70)

        all_exist = True
        for file_path in expected_files:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✅ {file_path} ({size:,} bytes)")
            else:
                print(f"❌ {file_path} - NOT FOUND")
                all_exist = False

        print("=" * 70)
        return all_exist

    def display_summary(self):
        """Display orchestrator run summary."""
        print("\n" + "=" * 70)
        print(f"📊 {self.domain_name.upper()} ORCHESTRATOR SUMMARY")
        print("=" * 70)
        print(f"Domain: {self.domain_name}")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print(f"Completed Phases: {len(self.completed_phases)}")
        print(f"Total Cost: ${self.total_cost:.4f}")

        if self.phase_costs:
            print("\nCost Breakdown:")
            for phase, cost in self.phase_costs.items():
                print(f"  - {phase}: ${cost:.4f}")

        print("=" * 70)

    @abstractmethod
    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        """Get agent definitions for this orchestrator.

        Returns:
            Dictionary mapping agent names to AgentDefinition objects
        """
        pass

    @abstractmethod
    def get_allowed_tools(self) -> List[str]:
        """Get list of allowed tools for this orchestrator.

        Returns:
            List of tool names
        """
        pass

    @abstractmethod
    async def run(self):
        """Run the orchestrator workflow.

        This method must be implemented by subclasses to define the
        specific workflow for their domain.
        """
        pass

    async def execute_phase(
        self,
        phase_name: str,
        agent_name: str,
        prompt: str,
        client: ClaudeSDKClient,
    ):
        """Execute a single phase of the workflow.

        Args:
            phase_name: Name of the phase
            agent_name: Name of the agent to use
            prompt: Prompt for the agent
            client: Claude SDK client
        """
        await client.query(prompt)

        phase_cost = 0.0
        async for msg in client.receive_response():
            self.display_message(msg)
            if isinstance(msg, ResultMessage) and msg.total_cost_usd:
                phase_cost = msg.total_cost_usd

        self.track_phase_cost(phase_name, phase_cost)
        self.mark_phase_complete(phase_name)

    def create_client_options(
        self,
        permission_mode: str = "acceptEdits",
        cwd: str = ".",
    ) -> ClaudeAgentOptions:
        """Create Claude SDK client options.

        Args:
            permission_mode: Permission mode for the client
            cwd: Current working directory

        Returns:
            ClaudeAgentOptions configured for this orchestrator
        """
        agents = self.get_agent_definitions()
        allowed_tools = self.get_allowed_tools()

        return ClaudeAgentOptions(
            agents=agents,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            cwd=cwd,
        )

    async def run_with_client(self):
        """Run the orchestrator with automatic client setup and teardown."""
        print("\n" + "=" * 70)
        print(f"🚀 {self.domain_name.upper()} Orchestrator")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"\nStarting {self.domain_name} analysis workflow...")
        print("=" * 70)

        options = self.create_client_options()

        try:
            async with ClaudeSDKClient(options=options) as client:
                # Store client for use in run()
                self.client = client
                await self.run()

            self.display_summary()

            print(f"\n✅ Analysis complete! View results in: {self.output_dir.absolute()}")
            return True

        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print(f"\nPartial results may be available in: {self.output_dir.absolute()}")
            raise
        finally:
            # Clean up
            if hasattr(self, 'client'):
                delattr(self, 'client')


class CrossOrchestratorCommunication:
    """Mixin for orchestrators that need to communicate with each other."""

    def __init__(self):
        """Initialize cross-orchestrator communication."""
        self.orchestrator_registry: Dict[str, BaseOrchestrator] = {}

    def register_orchestrator(self, name: str, orchestrator: BaseOrchestrator):
        """Register an orchestrator for cross-domain communication.

        Args:
            name: Name of the orchestrator
            orchestrator: Orchestrator instance
        """
        self.orchestrator_registry[name] = orchestrator

    async def invoke_orchestrator(
        self,
        orchestrator_name: str,
        phase_name: str,
        context: Dict[str, Any],
    ) -> Any:
        """Invoke another orchestrator for cross-domain validation.

        Args:
            orchestrator_name: Name of the orchestrator to invoke
            phase_name: Name of the phase to execute
            context: Context data to pass

        Returns:
            Result from the orchestrator
        """
        if orchestrator_name not in self.orchestrator_registry:
            raise ValueError(f"Orchestrator '{orchestrator_name}' not registered")

        orchestrator = self.orchestrator_registry[orchestrator_name]

        # This is a simplified version - would need more sophisticated
        # inter-orchestrator communication in production
        print(f"\n🔗 Cross-orchestrator call: {self.domain_name} → {orchestrator_name}")
        print(f"   Phase: {phase_name}")
        print(f"   Context: {list(context.keys())}")

        # Execute the requested phase with the provided context
        # Implementation would depend on specific orchestrator needs
        return {"status": "success", "message": f"Invoked {orchestrator_name}.{phase_name}"}
