# Concept Definition

## Output Format

**ONLY ONE heading is allowed**: `## What Is/Are [Technology Name]?`

**DO NOT** use additional Markdown headings (###, ####) or sub-sections within this
section. Write in continuous, flowing paragraphs with minimal structural markup.

## Content Structure

The section must follow this three-part structure:

### Part 1: Concept Definition (What)

- Define the technology clearly and concisely
- Use contrast: *"Unlike traditional X, this does Y"*
- Focus on **what class of problems** it is designed for
- Do NOT use negative framing like "What X is NOT"
- Keep to 1-2 paragraphs

### Part 2: Application Examples

- Reference concrete, real-world systems or applications
- Avoid hypothetical or future-only framing
- Link to actual products where possible
- Keep to 1 paragraph

### Part 3: Key Characteristics

- Summarize 4-6 core capabilities that define the technology
- Use a **bullet list** for clarity
- Each bullet describes **what the system can do**, not **how it is implemented**
- Focus on capabilities, not mechanisms

## Quality Bar

A reader should finish this section thinking:

> "I understand what this technology is, how it differs from existing approaches,
> and what real problems it can solve. The examples convince me it's real and
> practical."

## Reference Pattern (Deep Agents)

## What Are Deep Agents?

[Deep Agents](https://blog.langchain.com/deep-agents/) are an advanced agent architecture designed for handling complex, multi-step tasks that require sustained
reasoning, tool use, and memory. Unlike traditional agents that operate in short
loops or perform simple tool calls, Deep Agents plan their actions, manage evolving
context, delegate subtasks to specialized sub-agents, and maintain state across
long interactions. This architecture is already powering real-world applications
like [Claude Code](https://www.datacamp.com/tutorial/claude-code), [Deep Research](https://www.datacamp.com/blog/deep-research-openai), and [Manus](https://www.datacamp.com/tutorial/manus-ai).

These are the key characteristics of Deep Agents:

- **Planning capability**: They can break down large tasks into manageable
  subtasks and adjust the plan as work progresses.
- **Context management**: They retain and reference important information across
  long conversations and multiple steps.
- **Sub-agent delegation**: Deep Agents can launch specialized sub-agents to
  handle focused parts of a task.
- **File system integration**: They persist and retrieve information as needed,
  enabling true "memory" beyond a single conversation turn.
- **Detailed system prompts**: Deep Agents follow explicit workflows for
  consistency and reliability while operating with sophisticated instructions and
  examples.
