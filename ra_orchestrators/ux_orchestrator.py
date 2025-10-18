#!/usr/bin/env python3
"""UX/UI Design Orchestrator using Claude Agent SDK.

Comprehensive UX design workflow with:
- User research and persona development
- Information architecture design
- Visual design with Figma MCP integration
- Interactive prototyping
- API contract design
- Design system documentation
"""

from pathlib import Path
from typing import Dict, List

from claude_agent_sdk import AgentDefinition

from .base_orchestrator import BaseOrchestrator


class UXOrchestrator(BaseOrchestrator):
    """Orchestrator for comprehensive UX/UI design workflow."""

    def __init__(
        self,
        project_name: str = "Project",
        output_base_dir: Path = Path("ra_output"),
        show_tool_details: bool = True,
        use_timestamp: bool = True,
    ):
        """Initialize UX orchestrator.

        Args:
            project_name: Name of the project being designed
            output_base_dir: Base directory for design outputs (default: ra_output)
            show_tool_details: Whether to display detailed tool usage
            use_timestamp: Whether to append timestamp to output directory
        """
        super().__init__(
            domain_name="ux",
            output_base_dir=output_base_dir,
            show_tool_details=show_tool_details,
            use_timestamp=use_timestamp,
        )

        self.project_name = project_name

        # Define subdirectories for 6-phase workflow
        # self.output_dir is already set by BaseOrchestrator as ra_output/ux_{timestamp}/
        self.research_dir = self.output_dir / "01_research"
        self.ia_dir = self.output_dir / "02_ia"
        self.design_dir = self.output_dir / "03_design"
        self.prototypes_dir = self.output_dir / "04_prototypes"
        self.api_contracts_dir = self.output_dir / "05_api_contracts"
        self.design_system_dir = self.output_dir / "06_design_system"

        # Create directory structure
        self.create_output_structure()

    def create_output_structure(self, subdirs: List[str] = None):
        """Create output directory structure for UX workflow."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.ia_dir.mkdir(parents=True, exist_ok=True)
        self.design_dir.mkdir(parents=True, exist_ok=True)
        self.prototypes_dir.mkdir(parents=True, exist_ok=True)
        self.api_contracts_dir.mkdir(parents=True, exist_ok=True)
        self.design_system_dir.mkdir(parents=True, exist_ok=True)

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        """Get agent definitions for UX design workflow.

        Returns:
            Dictionary of agent definitions
        """
        ux_researcher = AgentDefinition(
            description="Conducts user research, creates personas, and analyzes user journeys",
            prompt="""You are a UX researcher expert. Your job is to:

1. Analyze project requirements and user needs
2. Create detailed user personas with demographics, goals, and pain points
3. Map user journeys and identify key touchpoints
4. Conduct competitive analysis and identify best practices
5. Synthesize research findings into actionable insights

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

Be data-driven but empathetic. Focus on understanding user needs deeply.""",
            tools=["Read", "Write", "Grep", "Glob", "WebSearch"],
            model="sonnet",
        )

        ia_architect = AgentDefinition(
            description="Designs information architecture, sitemaps, and navigation structures",
            prompt="""You are an information architecture expert. Your job is to:

1. Organize content into logical hierarchies
2. Create clear sitemaps showing page relationships
3. Design intuitive navigation structures
4. Define content models and taxonomies
5. Create wireframes showing page layouts and structure

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

Focus on clarity, findability, and logical organization.""",
            tools=["Read", "Write", "Grep", "Glob"],
            model="sonnet",
        )

        ui_designer = AgentDefinition(
            description="Creates visual designs and high-fidelity mockups",
            prompt="""You are a UI designer expert. Your job is to:

1. Create high-fidelity visual designs
2. Apply design principles (typography, color, spacing, hierarchy)
3. Ensure accessibility (WCAG compliance)
4. Design responsive layouts for multiple screen sizes
5. Document design specifications and guidelines

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

When Figma MCP tools are available, use them to create actual designs.
Otherwise, create detailed design specifications in markdown with Mermaid diagrams.

Focus on visual hierarchy, consistency, and user delight.""",
            tools=["Read", "Write", "Grep", "Glob"],
            model="sonnet",
        )

        prototype_developer = AgentDefinition(
            description="Creates interactive prototypes and validates user flows",
            prompt="""You are a prototyping expert. Your job is to:

1. Create interactive prototypes that demonstrate user flows
2. Validate designs through user testing scenarios
3. Document interaction patterns and micro-interactions
4. Create clickable prototypes (or detailed interaction specs)
5. Identify technical feasibility and constraints

IMPORTANT: When asked to write to a file, ALWAYS use the Write tool
to create the actual file. Do not just describe what you would write.

When design-to-code tools are available, generate working prototypes.
Otherwise, create detailed interaction specifications.

Focus on usability, feedback, and delightful interactions.""",
            tools=["Read", "Write", "Grep", "Glob", "Bash"],
            model="sonnet",
        )

        return {
            "ux-researcher": ux_researcher,
            "ia-architect": ia_architect,
            "ui-designer": ui_designer,
            "prototype-developer": prototype_developer,
        }

    def get_allowed_tools(self) -> List[str]:
        """Get list of allowed tools for UX design workflow.

        Returns:
            List of tool names
        """
        return ["Read", "Write", "Grep", "Glob", "Bash", "WebSearch"]

    async def phase_1_ux_research(self):
        """Phase 1: UX Research - Requirements, personas, user journeys."""
        self.display_phase_header(1, "UX Research", "🔍")

        await self.execute_phase(
            phase_name="UX Research",
            agent_name="ux-researcher",
            prompt=f"""Use the ux-researcher agent to conduct comprehensive user research for {self.project_name}.

Analyze and document:
1. Project requirements and business objectives
2. Target user demographics and characteristics
3. User personas (3-5 detailed personas)
4. User journey maps showing key touchpoints
5. Competitive analysis of similar products
6. Key insights and design opportunities

Write your research to: {self.research_dir}/user_research.md

Use this structure:
# UX Research: {self.project_name}

## Project Overview
[Requirements, objectives, success criteria]

## Target Users
[Demographics, characteristics, contexts of use]

## User Personas
### Persona 1: [Name]
- Demographics: [Age, role, tech comfort]
- Goals: [What they want to achieve]
- Pain Points: [Current frustrations]
- Motivations: [What drives them]

[Repeat for 3-5 personas]

## User Journey Maps
[Map critical user journeys with touchpoints, emotions, opportunities]

## Competitive Analysis
[Analysis of 3-5 competitors with strengths/weaknesses]

## Key Insights
[Actionable insights for design]

Be thorough and evidence-based. Use Mermaid diagrams for journey maps.""",
            client=self.client,
        )

    async def phase_2_information_architecture(self):
        """Phase 2: Information Architecture - Sitemaps, navigation, content structure."""
        self.display_phase_header(2, "Information Architecture", "🗺️")

        await self.execute_phase(
            phase_name="Information Architecture",
            agent_name="ia-architect",
            prompt=f"""Use the ia-architect agent to design the information architecture for {self.project_name}.

Based on the user research in {self.research_dir}/user_research.md, create:
1. Comprehensive sitemap showing all pages and hierarchy
2. Navigation structure (primary, secondary, utility nav)
3. Content models defining data structures
4. Wireframes for key pages showing layout and content placement
5. Search and filtering strategies (if applicable)

Write your architecture to: {self.ia_dir}/information_architecture.md

Use this structure:
# Information Architecture: {self.project_name}

## Sitemap
```mermaid
graph TD
    Home[Home]
    Home --> Section1[Section 1]
    Home --> Section2[Section 2]
    [Continue mapping all pages]
```

## Navigation Structure
### Primary Navigation
[Top-level navigation items]

### Secondary Navigation
[Section-specific navigation]

### Utility Navigation
[Account, settings, help, etc.]

## Content Models
### [Entity Name]
- Field 1: [Type, description]
- Field 2: [Type, description]

## Wireframes
### Homepage
[ASCII or Mermaid diagram showing layout]

### [Key Page 2]
[Layout diagram]

[Continue for 5-8 key pages]

Use Mermaid diagrams for sitemaps and flowcharts.""",
            client=self.client,
        )

    async def phase_3_visual_design(self):
        """Phase 3: Visual Design - High-fidelity mockups and design specs."""
        self.display_phase_header(3, "Visual Design", "🎨")

        await self.execute_phase(
            phase_name="Visual Design",
            agent_name="ui-designer",
            prompt=f"""Use the ui-designer agent to create visual designs for {self.project_name}.

Based on the information architecture in {self.ia_dir}/information_architecture.md, create:
1. Visual design system (colors, typography, spacing, components)
2. High-fidelity mockups for key pages
3. Responsive design specifications (mobile, tablet, desktop)
4. Accessibility guidelines (WCAG compliance)
5. Design specifications for developers

NOTE: If Figma MCP tools are available, use them to create actual designs.
Otherwise, create detailed design specifications with examples.

Write your designs to: {self.design_dir}/visual_design.md

Use this structure:
# Visual Design: {self.project_name}

## Design System

### Color Palette
- Primary: #XXXXXX (usage)
- Secondary: #XXXXXX (usage)
- Accent: #XXXXXX (usage)
- Neutrals: #XXXXXX, #XXXXXX, #XXXXXX
- Semantic: Success, Warning, Error, Info

### Typography
- Headings: [Font family, sizes, weights]
- Body: [Font family, sizes, line heights]
- Scale: H1, H2, H3, Body, Small, Caption

### Spacing System
- Base unit: [8px, 4px, etc.]
- Scale: 4, 8, 16, 24, 32, 48, 64, 96px

### Component Library
[List of reusable components with variants]

## High-Fidelity Mockups

### Homepage
[Detailed description or Figma link]
- Layout structure
- Visual hierarchy
- Interactive elements
- Responsive behavior

### [Key Page 2]
[Detailed mockup description]

[Continue for 5-8 key pages]

## Responsive Design
[Breakpoints and responsive behaviors]

## Accessibility
[WCAG compliance guidelines, contrast ratios, focus states]

Include visual examples using code blocks or Mermaid diagrams.""",
            client=self.client,
        )

    async def phase_4_interactive_prototyping(self):
        """Phase 4: Interactive Prototyping - Working prototypes and interaction specs."""
        self.display_phase_header(4, "Interactive Prototyping", "⚡")

        await self.execute_phase(
            phase_name="Interactive Prototyping",
            agent_name="prototype-developer",
            prompt=f"""Use the prototype-developer agent to create interactive prototypes for {self.project_name}.

Based on the visual designs in {self.design_dir}/visual_design.md, create:
1. Interactive prototype specifications
2. User flow demonstrations for critical paths
3. Micro-interaction details
4. Animation and transition specifications
5. Usability testing scenarios

NOTE: If design-to-code tools are available, generate working prototypes.
Otherwise, create detailed interaction specifications.

Write your prototypes to: {self.prototypes_dir}/interactive_prototypes.md

Use this structure:
# Interactive Prototypes: {self.project_name}

## Critical User Flows

### Flow 1: [Primary user journey]
```mermaid
sequenceDiagram
    participant User
    participant UI
    participant System
    [Show interaction sequence]
```

**Interactions:**
1. [Step-by-step interaction details]
2. [Include feedback, validation, error states]

[Continue for 3-5 critical flows]

## Micro-Interactions

### [Component Name]
- Trigger: [What initiates the interaction]
- Feedback: [Visual/audio feedback]
- Duration: [Animation timing]
- Easing: [Animation curve]

## Animation Specifications
[Define key animations with timing and easing]

## Interactive States
### [Component Name]
- Default state
- Hover state
- Active/pressed state
- Focused state
- Disabled state
- Loading state
- Error state

## Usability Testing Scenarios
[Define 3-5 testing scenarios with success criteria]

Use Mermaid sequence diagrams for user flows.""",
            client=self.client,
        )

    async def phase_5_api_contract_design(self):
        """Phase 5: API Contract Design - Frontend-backend interface specifications."""
        self.display_phase_header(5, "API Contract Design", "🔌")

        await self.execute_phase(
            phase_name="API Contract Design",
            agent_name="ia-architect",
            prompt=f"""Use the ia-architect agent to design API contracts for {self.project_name}.

Based on the prototypes in {self.prototypes_dir}/interactive_prototypes.md, define:
1. Data models and schemas
2. API endpoints for each user flow
3. Request/response specifications
4. Error handling and validation
5. Real-time data requirements (if applicable)

Write your API contracts to: {self.api_contracts_dir}/api_specifications.md

Use this structure:
# API Contract Specifications: {self.project_name}

## Data Models

### [Entity Name]
```json
{{
  "id": "string (UUID)",
  "field1": "type",
  "field2": "type",
  "createdAt": "datetime",
  "updatedAt": "datetime"
}}
```

[Continue for all entities]

## API Endpoints

### User Flow: [Flow Name]

#### GET /api/[resource]
**Purpose:** [What this endpoint does]

**Request:**
- Headers: Authorization, Content-Type
- Query params: page, limit, filter

**Response (200):**
```json
{{
  "data": [],
  "meta": {{"total": 0, "page": 1}}
}}
```

**Error Responses:**
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Server Error

[Continue for all endpoints]

## Real-Time Requirements
[WebSocket/SSE specifications if needed]

## Validation Rules
[Frontend and backend validation requirements]

## Error Handling Strategy
[How errors should be handled and displayed]

Use JSON examples for clarity.""",
            client=self.client,
        )

    async def phase_6_design_system_documentation(self):
        """Phase 6: Design System Documentation - Component library and style guide."""
        self.display_phase_header(6, "Design System Documentation", "📐")

        await self.execute_phase(
            phase_name="Design System Documentation",
            agent_name="ui-designer",
            prompt=f"""Use the ui-designer agent to create comprehensive design system documentation for {self.project_name}.

Based on all previous work, synthesize:
1. Complete component library
2. Design tokens and variables
3. Usage guidelines and best practices
4. Code examples for developers
5. Contribution guidelines

Review these documents:
- {self.design_dir}/visual_design.md
- {self.prototypes_dir}/interactive_prototypes.md
- {self.api_contracts_dir}/api_specifications.md

Write your design system to: {self.design_system_dir}/design_system.md

Use this structure:
# Design System: {self.project_name}

## Overview
[Purpose, principles, and scope of the design system]

## Design Tokens

### Colors
```css
--color-primary: #XXXXXX;
--color-secondary: #XXXXXX;
[All color tokens]
```

### Typography
```css
--font-family-primary: 'Font Name', sans-serif;
--font-size-base: 16px;
[All typography tokens]
```

### Spacing
```css
--spacing-unit: 8px;
--spacing-xs: 4px;
[All spacing tokens]
```

## Component Library

### Button
**Variants:** Primary, Secondary, Tertiary, Ghost, Danger

**States:** Default, Hover, Active, Focus, Disabled, Loading

**Props:**
- size: sm | md | lg
- variant: primary | secondary | ...
- disabled: boolean
- loading: boolean

**Usage:**
```jsx
<Button variant="primary" size="md">
  Click me
</Button>
```

**Accessibility:**
- ARIA labels
- Keyboard navigation
- Focus management

[Continue for all components: Input, Dropdown, Modal, Card, etc.]

## Usage Guidelines

### Do's
✅ [Best practices]

### Don'ts
❌ [Common mistakes to avoid]

## Implementation Guide
[Technical guidance for developers]

## Contribution Guidelines
[How to extend the design system]

Make it comprehensive and developer-friendly.""",
            client=self.client,
        )

    async def run(self):
        """Run comprehensive UX design workflow in 6 phases."""
        # Run all UX design phases
        await self.phase_1_ux_research()
        await self.phase_2_information_architecture()
        await self.phase_3_visual_design()
        await self.phase_4_interactive_prototyping()
        await self.phase_5_api_contract_design()
        await self.phase_6_design_system_documentation()

        # Verify all outputs
        expected_files = [
            self.research_dir / "user_research.md",
            self.ia_dir / "information_architecture.md",
            self.design_dir / "visual_design.md",
            self.prototypes_dir / "interactive_prototypes.md",
            self.api_contracts_dir / "api_specifications.md",
            self.design_system_dir / "design_system.md",
        ]

        await self.verify_outputs(expected_files)


async def main():
    """Run comprehensive UX design workflow."""
    import sys

    project_name = sys.argv[1] if len(sys.argv) > 1 else "Sample Project"
    orchestrator = UXOrchestrator(project_name=project_name)
    await orchestrator.run_with_client()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
