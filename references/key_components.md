# Key Components

## Output Format

The section heading should be: `## Key Components of [Technology Name]`

Write each component as a natural, flowing narrative — not as structured
subsections. The explanation should integrate purpose, context, and code examples
organically.

## Content Structure

### Opening Paragraph

Start with 1-2 sentences that:
- Introduce how many core components exist
- Explain how they work together
- Set the context for what follows

Example: "A RAG system consists of five core components that work together to
transform raw documents into grounded, citable answers."

### Component Sections

For each component, use this format:

#### 1. Component Name (### heading)

Write 2-4 paragraphs that naturally integrate:

1. **What it does** — Describe the component's purpose in plain language
2. **Why it's needed** — Explain the problem it solves
3. **Code example** — Include minimal, concrete code that demonstrates the
   component in action
4. **How it works** — Briefly explain the example or the component's behavior

**Writing guidelines:**
- Start with a clear problem statement or purpose
- Embed the code example naturally within the explanation
- Avoid bullet lists for responsibilities — write in prose
- Keep code snippets minimal (5-15 lines)
- Use pseudocode or small config snippets; avoid full implementations
- Include just enough code to make the component concrete
- Focus on **what** and **why**, not just implementation details

#### Separator

Use `---` between components for visual separation.

### Interaction & Data Flow

After covering all components, add a section explaining:
- How components communicate
- What information flows between them
- Where state lives (memory, files, database, etc.)
- **Include a visual flow diagram** — Use ASCII art to show the pipeline phases and data flow

Heading: `## Interaction & Data Flow`

**Diagram Guidelines:**

When applicable, include a visual flow diagram using ASCII box-drawing characters. Show the distinct phases (e.g., offline vs online, indexing vs retrieval) and how data flows between components.

Example structure:
```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE NAME                                  │
│                        (timing/trigger)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input ──► Step 1 ──► Step 2 ──► Output                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

For systems with persistent state, also include a **State Management** table:

| State | Where It Lives | Update Frequency |
|-------|----------------|------------------|
| Data X | Database | On change |
| Data Y | Memory | Per request |

## Quality Bar

A reader should finish this section thinking:

> "I understand what each component does, why it's necessary, and how they work
> together. The code examples make it concrete without being overwhelming."

## Reference Pattern (Deep Agents)

## Core Components of Deep Agents

Deep Agents overcome the limitations of traditional agents through four core
components:

### 1. Detailed system prompts

Unlike simple instruction prompts, Deep Agents use comprehensive system prompts
that integrate planning, research, and delegation with documentation, utilizing
few-shot examples to decompose complex tasks:

```python
DEEP_AGENT_SYSTEM_PROMPT = """
You are a specialized AI agent designed to handle complex, multi-step tasks.

Your workflow:
1. Break down the task into subtasks
2. Execute each subtask systematically
3. Delegate specialized work to sub-agents when needed
4. Maintain context across all operations

Examples of effective task decomposition:
...
"""
```

This detailed prompt structure ensures consistent behavior while providing the
agent with explicit guidance on handling complexity.

### 2. Planning tools

The planning tool is often a simple but effective mechanism that helps the agent
organize its thoughts and maintain visibility into its progress:

```python
@tool
def todo_write(tasks: List[str]) -> str:
    """Create or update a todo list for tracking task progress."""
    formatted_tasks = "\n".join([f"- {task}" for task in tasks])
    return f"Todo list created:\n{formatted_tasks}"
```

This simple tool provides important context engineering, which forces the agent
to plan accordingly and keep that plan visible throughout execution.

### 3. Sub-agent delegation

Deep Agents can launch specialized sub-agents to handle focused parts of a task,
allowing parallel work and domain-specific expertise:

```python
def launch_sub_agent(task: str, expertise: str) -> str:
    """Launch a specialized sub-agent for a specific task type."""
    agent = create_agent(
        system_prompt=f"You are an expert in {expertise}.",
        tools=get_tools_for_domain(expertise)
    )
    return agent.run(task)
```

Rather than handling every aspect of a complex task itself, the main agent
delegates to specialists — similar to how a project manager delegates to team
members with different skills.

### 4. File system integration

Deep Agents persist and retrieve information as needed, enabling true "memory"
beyond a single conversation turn:

```python
@tool
def save_context(key: str, value: str) -> str:
    """Save information to persistent storage for later retrieval."""
    with open(f"./context/{key}.txt", "w") as f:
        f.write(value)
    return f"Saved context to {key}.txt"

@tool
def load_context(key: str) -> str:
    """Load previously saved information."""
    with open(f"./context/{key}.txt", "r") as f:
        return f.read()
```

This persistent memory allows agents to maintain state across sessions and build
knowledge over time, rather than starting fresh each time.



