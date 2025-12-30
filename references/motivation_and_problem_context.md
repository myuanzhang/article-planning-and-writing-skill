# Motivation & Problem Context

## Output Format

**DO NOT** output a section heading like `## Motivation & Problem Context`. The
generated section should start directly with natural paragraph content.

The section must be written in **natural paragraph format**, divided into 2-4
paragraphs following the structure below. Use the reference pattern as your
primary guide for style and tone.

## Content Structure

### Paragraph 1: Problem Background

- **What is the dominant or common approach today?**
  - Describe the current mainstream solution or architecture in 1-2 sentences.
- **Where does it fail in practice?**
  - Focus on concrete failure modes (e.g., scale, complexity, time horizon,
    cost, latency).
- **Why are these failures fundamental, not incidental?**
  - Tie failures to structural constraints (e.g., context limits, architectural
    limitations, missing capabilities).

Use a conversational but technical tone. Start with the status quo, then reveal
its limitations progressively.

### Paragraph 2: Solution Provided

- Describe how this technology addresses the problems mentioned.
- Reference real applications, adoption pressure, or recent shifts that make the
  solution relevant now.
- Mention real-world examples or proven use cases if available.

### Paragraph 3: Tutorial Goals (Optional but Recommended)

- Explicitly state what the reader will learn or be able to do after following
  this tutorial.
- Use a bullet list for clarity (3-5 items).
- Keep outcomes concrete and actionable.

## Quality Bar

A reader should finish this section thinking:

> "Yes, I've hit this problem before — and I see why current tools can't really
> solve it. This new approach makes sense, and I can see myself using it."

## Reference Pattern (Deep Agents)

The most common agent architecture today involves an LLM calling tools in a loop,
which is simple, effective, but ultimately limited. While this approach works for
straightforward tasks, it falls short when faced with complex, multi-step
challenges that require planning, context management, and sustained execution
over longer time horizons.

LangChain's Deep Agents architecture combines detailed system prompts, planning
tools, sub-agents, and file systems to create AI agents capable of tackling
complex research, coding, and analytical tasks. Applications like Claude Code,
Deep Research, and Manus have proven this approach's effectiveness, and now the
deepagents Python package makes this architecture accessible to everyone.

In this tutorial, I'll explain step by step how to:

- Build Deep Agents that handle complex workflows and manage context effectively
- Create a job application assistant that searches for positions and generates
  tailored cover letters
- Implement specialized sub-agents for focused task execution and context
  management

