---
name: article-planning-and-writing
description: Design and generate end-to-end engineering tutorials that introduce a complex system or framework, decompose its core architecture, and walk through a reproducible, real-world demo from setup to production-style output. Use this Skill when documenting how a system actually works in practice, not just what it is.
---


## Overview

This Skill is designed for engineers and builders who need to understand **how a system actually works in practice**, not just its surface concepts.

It enforces a structured learning flow:

**Why → What → How → Demo → Reflection**

Each tutorial produced with this Skill explains:

* **Why** the system exists and what real problems it solves
* **What** its core concepts and architectural components are
* **How** those components work together
* **How to build** a reproducible, real-world demo that reflects production constraints
* **When to use** the approach—and when not to

The goal is to deliver **engineering-grade tutorials** that move from concept to implementation, not conceptual overviews or trend discussions.

## Execution Flow（MANDATORY）

When generating a technical tutorial using this Skill, follow this workflow strictly:

### Phase 1: Article Planning

1. Based on the given topic, **analyze its technical scope and practical depth**.
2. Load `article_outline.md` as the **structural template**.
3. Generate a **complete, section-level outline** tailored to the topic.
4. Ensure the outline reflects a full **Why → What → How → Demo → Reflection** progression.

> Output: A finalized article outline only.
> Do NOT write any article content in this phase.

------

### Phase 2: Section-by-Section Writing

For EACH section in the outline:

1. Load the corresponding `references/<section>.md`.
2. Read and follow **all writing rules and constraints** in that reference file.
3. Write **ONLY the current section**.
4. Do NOT reference future sections.
5. Do NOT summarize or merge content across sections.

> Each section is treated as an **independent writing operation**.

------

### Phase 3: Assembly, Consistency Check, and Save

After ALL sections are completed:

1. Review the article end-to-end for:
   - Terminology consistency
   - Architectural coherence
   - Demo and code alignment
   - Logical flow between sections
2. Ensure the final article reads as a **single, cohesive engineering tutorial**.
3. Do NOT introduce new concepts, examples, or code at this stage.
4. **Save the article** using the Write tool:
   - Generate a filename based on the topic (e.g., `rag_tutorial.md`, `deep_agents_guide.md`)
   - Save to the current working directory or a user-specified path
   - Confirm the save location with the user before writing
