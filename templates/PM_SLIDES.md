# Slide 1 — Title & Hook
- Product name, one-line value prop
- Hook: baseline pain and desired outcome

# Slide 2 — Problem
- Single-sentence problem
- Scope, non-goals, baseline metrics

# Slide 3 — Success
- North Star + guardrails
- Measurement plan

# Slide 4 — Audience
- Persona & JTBD
- Top tasks and acceptance criteria
- Stakeholder RACI (mini)

# Slide 5 — Solution Overview
- Before/after flow
- Where AI intervenes & why

**LLM Stack → 2025 Lifecycle (Layer 0–8):**

```mermaid
flowchart TB
  %% AI Product Management: Evolved LLM App Stack (2025)
  %% Linear backbone with lifecycle loop and governance overlay

  %% Layer 0: Foundation
  subgraph L0[Layer 0: Context & Provenance]
    direction TB
    L0a[Problem & Success Definitions]
    L0b[Personas & JTBD]
    L0c[Data Lineage & Consent]
  end

  %% Data Layer (L1-L3)
  subgraph DataLayer[" "]
    direction LR
    L1["Layer 1: Data Pipelines<br/><small>ingest • transform • contracts</small>"]
    L2["Layer 2: Embedding Models<br/><small>domain semantics • drift checks</small>"]
    L3["Layer 3: Vector DB<br/><small>hybrid retrieval: graph + vector</small>"]
  end

  %% Processing Layer (L4-L5)
  subgraph ProcessingLayer[" "]
    direction LR
    L4["Layer 4: Orchestrators / Agents<br/><small>tool use • autonomy bounds • MCP/A2A</small>"]
    L5["Layer 5: UX<br/><small>citations • confidence • overrides</small>"]
  end

  %% Operations Layer (L6-L7)
  subgraph OpsLayer[" "]
    direction LR
    L6["Layer 6: Evaluation & Observability<br/><small>quality • cost • latency • safety</small>"]
    L7["Layer 7: Lifecycle & Ops<br/><small>versioning • rollbacks • flags • budgets</small>"]
  end

  %% Governance Layer
  L8["Layer 8: Governance & Feedback Loop<br/><small>PII • audit • fairness • HIL</small>"]

  %% Main flow
  L0 --> DataLayer
  DataLayer --> ProcessingLayer
  ProcessingLayer --> OpsLayer
  OpsLayer --> L8

  %% Data flow within layers
  L1 --> L2 --> L3
  L4 --> L5
  L6 --> L7

  %% Feedback loops
  L8 -.->|"continuous feedback"| L0
  L6 -.->|"metrics & insights"| L0

  %% Styling
  classDef foundation fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20
  classDef dataLayer fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#0d47a1
  classDef processingLayer fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
  classDef opsLayer fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c
  classDef govLayer fill:#ffebee,stroke:#f44336,stroke-width:3px,stroke-dasharray:5 5,color:#b71c1c

  class L0,L0a,L0b,L0c foundation
  class L1,L2,L3 dataLayer
  class L4,L5 processingLayer
  class L6,L7 opsLayer
  class L8 govLayer
```

# Slide 6 — Demo
- Storyboard frames
- Trust, control, feedback loop
- Experiment plan

# Slide 7 — Infra Diagram
- Data → Decision → UX
- Governance overlays, monitoring, rollbacks

**Compact lifecycle view (for Infra focus):**
```mermaid
flowchart TB
  %% Compact lifecycle view (for Infra focus)
  
  subgraph Foundation[" "]
    L0[Layer 0: Context]
  end

  subgraph DataFlow["Data Flow"]
    direction LR
    L1[Data]
    L2[Embeddings]
    L3[Vector DB]
  end

  subgraph Processing["Processing"]
    direction LR
    L4[Orchestrators/Agents]
    L5[UX]
  end

  subgraph Operations["Operations"]
    direction LR
    L6[Eval/Observability]
    L7[Lifecycle/Ops]
  end

  subgraph Governance[" "]
    L8[Governance/Feedback]
  end

  %% Main flow
  Foundation --> DataFlow
  DataFlow --> Processing
  Processing --> Operations
  Operations --> Governance

  %% Internal flows
  L1 --> L2 --> L3
  L4 --> L5
  L6 --> L7

  %% Feedback loop
  Governance -.->|"feedback"| Foundation

  %% Styling
  classDef foundation fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20
  classDef data fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#0d47a1
  classDef processing fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
  classDef ops fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c
  classDef gov fill:#ffebee,stroke:#f44336,stroke-width:2px,stroke-dasharray:5 5,color:#b71c1c

  class L0 foundation
  class L1,L2,L3 data
  class L4,L5 processing
  class L6,L7 ops
  class L8 gov
```
# Slide 8 — Results & Trade-offs
- Early signals / pilot data
- Key trade-offs & rationale

# Slide 9 — Conclusions
- Risks & mitigations
- Next steps
- GTM/enablement plan
