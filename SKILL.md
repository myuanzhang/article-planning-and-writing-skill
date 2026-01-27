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

---

## Content Accuracy & Currency Requirements (CRITICAL)

**All output must be verifiable and current at the time of publication.**

### Core Principles

1. **No guessing** — Never write code, API usage, or technical claims that you're not certain about
2. **Verify before writing** — Use web search to confirm any time-sensitive technical information
3. **Official sources only** — Prioritize official documentation, GitHub repos, and authoritative sources
4. **Explicit uncertainty** — If something cannot be verified, either omit it or explicitly state the uncertainty

### What Must Be Verified

Before including any of the following, verify via web search from official sources:

| Category | What to Check | How to Verify |
|----------|---------------|---------------|
| Package names | Current import paths | PyPI, npmjs.com, official docs |
| API syntax | Function signatures, parameters | Official API documentation |
| Version info | Latest stable versions | Package repositories, release notes |
| Deprecations | Deprecated features | Official deprecation notices |
| Installation | pip/npm install commands | Official installation guides |
| Best practices | Current recommended patterns | Official docs, recent blog posts |

### Verification Checklist

Before finalizing any article, verify:

- [ ] All package names are current and installable
- [ ] All import statements match current package structure
- [ ] All API calls match current documentation
- [ ] No deprecated features or syntax is used
- [ ] Version-specific claims are accurate
- [ ] Installation commands work as written
- [ ] External links/references still exist

### When to Use Web Search

**Mandatory search scenarios:**
- Writing code examples involving any library or framework
- Mentioning specific versions (e.g., "LangChain v0.1")
- Referencing API endpoints or function signatures
- Citing "latest" or "current" features
- Comparing technologies or approaches
- Referencing recent developments or announcements

**Search strategy:**
1. Start with official documentation (search: `[tech] official docs`)
2. Verify current stable version (search: `[package] latest version`)
3. Check for breaking changes (search: `[package] changelog breaking changes`)
4. Confirm API syntax (search: `[package] [function] API reference`)

### Handling Uncertainty

If verification fails or information is ambiguous:

| Situation | Action |
|-----------|--------|
| Package renamed | Use current name, mention old name in notes |
| API changed | Use current API, note version requirements |
| Version unclear | Omit specific version, use "latest stable" |
| Official docs unclear | Choose a simpler, clearer example |
| Cannot verify | Remove the content entirely |

### Self-Check Before Output

Before delivering final content, ask:

1. Did I verify all package names and imports?
2. Did I confirm all API syntax is current?
3. Did I check for deprecations or breaking changes?
4. Are all installation commands verifiable?
5. Is any time-sensitive information marked with dates?
6. Would this tutorial work if someone followed it today?

---

## Execution Flow（MANDATORY）

When generating a technical tutorial using this Skill, follow this workflow strictly:

### Phase 1: Article Planning

1. Based on the given topic, **analyze its technical scope and practical depth**.
2. Load `article_outline.md` as the **structural template**.
3. Generate a **section-level outline** tailored to the topic.
4. Ensure the outline reflects a full **Why → What → How → Demo → Reflection** progression.

> **CRITICAL OUTPUT FORMAT for Phase 1:**
> The outline MUST contain ONLY:
> - Title and one-line value proposition
> - Section headings (customized for the topic)
> - 2-3 sentence description per section explaining what will be covered
> - Demo step titles (not full step content)
>
> The outline MUST NOT contain:
> - Complete paragraphs
> - Full code examples
> - Detailed explanations
>
> This creates a clear boundary between Phase 1 (structure) and Phase 2 (content).
> **After completing the outline, immediately proceed to Phase 2 without user interaction.**

------

### Phase 2: Section-by-Section Writing

**CRITICAL:** This phase involves writing content internally ONLY. Do NOT output or display the article content to the user during this phase. Continue through ALL sections automatically, then proceed immediately to Phase 3.

For EACH section in the outline, in order:

1. Load the corresponding `references/<section>.md`.
2. Read and follow **all writing rules and constraints** in that reference file.
3. **Verify any time-sensitive technical information** using web search before writing
4. Write the current section content (internally, accumulate for Phase 3).
5. Immediately proceed to the next section.
6. Do not reference future sections while writing.
7. Do NOT summarize or merge content across sections.
8. **Do NOT output article content to user yet** — wait until Phase 3 when file is saved.

> Each section is treated as an **independent writing operation**, but all sections must be completed in one continuous session before moving to Phase 3.
> **After completing all sections, immediately transition to Phase 3 without user interaction.**

------

### Phase 3: Assembly, Consistency Check, and Save

After ALL sections are completed:

1. Review the article end-to-end for:
   - Terminology consistency
   - Architectural coherence
   - Demo and code alignment
   - Logical flow between sections
2. **Final accuracy verification** — Confirm all technical claims, code, and references are current
3. Ensure the final article reads as a **single, cohesive engineering tutorial**.
4. Do NOT introduce new concepts, examples, or code at this stage.
5. **Save the article** using the Write tool:
   - Generate a filename based on the topic (e.g., `rag_tutorial.md`, `deep_agents_guide.md`)
   - Save to the current working directory or a user-specified path
   - Confirm the save location with the user before writing
