# End-to-End Demo

## Output Format

The section heading should clearly indicate this is a demo/implementation section.
Use a format that fits your content — some options:

- `## End-to-End Demo: [Brief Description]`
- `## Demo: [Brief Description]`
- `## Hands-On: [Brief Description]`
- `## Building [X] From Scratch`

Example: `## End-to-End Demo: Building a Technical Documentation Assistant`

## Purpose

This section demonstrates **how the architecture works in practice** by walking
through a **complete, reproducible, real-world demo**.

The demo is not illustrative fluff — it is the **proof** that the concepts and
architecture described earlier are usable in production-like scenarios.

## Content Structure

### Opening Paragraph

Start with 1-2 sentences that:
- State what you're building
- Explain why this demo matters
- Set expectations for what readers will accomplish

Example: "In this section, I'll walk you through building a complete RAG system
from scratch. Our demo will create a **technical documentation assistant** that
can answer questions about a codebase or product documentation with source
citations."

### What We're Building (Optional)

Briefly list (2-4 bullets) what the demo will produce:
- The core functionality
- Key features or capabilities
- Measurable outcomes

### Step-by-Step Execution

Break the demo into **numbered steps**. Write each step as a natural narrative
that weaves together explanation and code — not as structured subsections.

Each step should include:
1. **Step heading** — `### Step N: [Action Name]`
2. **Explanation** — 1-2 sentences setting context for what you're about to do
3. **Code** — Minimal, reproducible code embedded naturally
4. **Follow-up explanation** — Brief description of what the code does or why it matters

Avoid rigid labels like `**Component**:` or `**Artifact Produced**:`. Instead,
integrate this information into flowing prose.

Use `---` between steps for visual separation.

### Complete Code File (REQUIRED)

**CRITICAL:** At the end of the demo section, you MUST provide a **complete, runnable code file** that combines all the code snippets from the previous steps. This file should:

- Be a single, complete file that readers can copy and run directly
- Include all necessary imports and dependencies
- Combine all code from individual steps into a working whole
- Be properly formatted and ready to execute
- Use consistent indentation and structure

**Format:**

```markdown
### Complete Code

Here's the complete `main.py` file that combines everything from the steps above:

```python
# main.py
# Complete working implementation

import ...

# All code from steps 1-N, combined into one runnable file
```

**Save this as `main.py` and run with:** `python main.py`
```

This complete file is essential because:
- Readers can verify their implementation
- No ambiguity about how pieces fit together
- Immediate "it works" feedback
- Reference for understanding the whole system

### Complete Example Output (Optional but Recommended)

Show what the final system produces when run:
- Terminal output
- File structure
- API responses
- UI screenshots (if applicable)

This helps readers verify they've built the system correctly.

## Demo Design Principles

Before writing the demo, ensure it satisfies all of the following:

- Uses a **real-world problem**, not a toy example
- Covers the **entire workflow**, not isolated snippets
- Produces **observable artifacts** (files, UI, reports, APIs)
- Explicitly maps steps to **architecture components**
- Can be reproduced with reasonable effort

> If a step cannot be reproduced, it does not belong in this section.

## Dependency & Version Management

**CRITICAL:** All code examples, API calls, package versions, and configuration syntax MUST be current and functional at the time of writing. Broken code wastes readers' time and damages trust.

> Tutorials with broken or outdated code are worse than no tutorial at all — they waste readers' time and damage trust.

### Mandatory Verification Steps

For each technology used in the demo, you **must** verify using web search from official sources:

1. **Verify current package names** — Search: `[package] official documentation`
   - Packages often rename or restructure (e.g., `openai` vs `openai-chat`)
   - Import paths change between major versions

2. **Verify current API syntax** — Search: `[package] [function] API reference`
   - Function signatures change
   - Parameter names and order may shift
   - Return types can differ

3. **Verify installation commands** — Search: `[package] installation guide`
   - Package names in pip/npm may differ from import names
   - Some packages require system dependencies

4. **Check for deprecations** — Search: `[package] deprecation notice` or `[package] migration guide`
   - Features marked deprecated should be avoided
   - Migration guides indicate breaking changes

5. **Verify version requirements** — Search: `[package] latest stable version`
   - Only pin versions when necessary for compatibility
   - When pinning, note the verification date

### Writing Guidelines

- **Prefer latest stable** — Let readers install the current stable version:
  ```bash
  pip install package-name  # not package-name==1.2.3
```

- **Pin only when necessary** — Pin versions only when:
  - The tutorial depends on specific behavior
  - Breaking changes exist between versions
  - Beta/unstable features are required

- **Document verification** — When pinning versions or using specific APIs:
  ```bash
  # Verified January 2026 from official docs
  # Requires langchain >= 0.3.0
  pip install langchain
  ```

- **Use official sources** — Link to authoritative documentation:
  - Python: PyPI, python.org, official GitHub repo
  - Node.js: npmjs.com, official docs
  - Rust: crates.io, docs.rs
  - Go: pkg.go.dev, official documentation

### Quality Checklist

Before finalizing the demo, confirm:

- [ ] All package names verified from official sources
- [ ] All import statements match current package structure
- [ ] All API calls verified against current documentation
- [ ] No deprecated features or syntax used
- [ ] Installation commands tested and working
- [ ] Version-specific claims are accurate
- [ ] External dependencies are accessible and current
- [ ] Code tested and runs without errors
- [ ] **A complete, runnable code file is provided at the end of the demo section**

### Search Strategy Examples

| Need | Search Query | Expected Source |
|------|--------------|-----------------|
| Current API | `langchain agents API reference` | python.langchain.com |
| Package name | `duckduckgo-search langchain install` | pypi.org or GitHub |
| Breaking changes | `langchain 0.1 to 0.2 migration` | LangChain blog/docs |
| Deprecations | `openai python deprecation list` | GitHub README or docs |
## Quality Bar

A reader should finish this section thinking:

> "I understand how to build this. The code works, I can run it myself, and I
> see how each part connects to the architecture described earlier."

## Technology & Language Choice

Use the technology stack that best fits the topic — no language restriction. Choose what is most relevant to the technology, widely used in its ecosystem, and accessible to the target audience. The reference pattern below uses Python for illustration; adapt the structure to your chosen language.

---

## Reference Pattern (Deep Agents)

## End-to-End Demo: Building a Job Application Assistant

In this section, I'll walk you through building a complete Deep Agent from scratch.
Our demo will create a **job application assistant** that searches for positions
and generates tailored cover letters.

**What we're building:**

- A Deep Agent that plans job application tasks
- Automated job search using web scraping
- Personalized cover letter generation
- File-based memory for tracking applications

---

### Step 1: Project Setup & Configuration

First, install the dependencies:

```bash
pip install deepagents langchain openai beautifulsoup4
```

Then create a project structure:

```bash
mkdir job-agent && cd job-agent
mkdir -p data output
touch .env requirements.txt main.py
```

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your-key-here
```

This sets up the basic project directory with `data/` for raw inputs like resumes
and job listings, and `output/` for the generated cover letters.

---

### Step 2: System Prompt Design

Define the agent's behavior and workflow in a comprehensive system prompt:

```python
SYSTEM_PROMPT = """
You are a job application assistant. Your workflow:

1. Search for jobs matching user criteria
2. Extract key requirements from each listing
3. Generate tailored cover letters using the user's resume
4. Save applications to the output folder

Always cite specific job requirements in your cover letters.
"""
```

The system prompt serves as the agent's "brain," encoding the entire workflow
so it can plan and execute autonomously without constant human guidance.

---

### Step 3: Planning Tool Integration

Add a todo list tool for tracking progress:

```python
from deepagents import tool

@tool
def update_todo_list(items: list[str]) -> str:
    """Update the task list for tracking job applications."""
    formatted = "\n".join(f"- {item}" for item in items)
    print(f"Current tasks:\n{formatted}")
    return formatted
```

This simple tool provides important context engineering — it forces the agent
to plan explicitly and keeps that plan visible throughout execution, making the
process more transparent and debuggable.

---

### Step 4: Sub-Agent Delegation

Create specialized sub-agents for search and writing tasks:

```python
def create_search_agent():
    return Agent(
        name="searcher",
        tools=[web_search, scrape_page],
        prompt="Find jobs matching criteria and extract requirements."
    )

def create_writer_agent():
    return Agent(
        name="writer",
        tools=[read_resume, generate_letter],
        prompt="Write tailored cover letters based on job requirements."
    )
```

Rather than handling every aspect itself, the main agent delegates to
specialists — similar to how a project manager assigns tasks to team members
with different expertise. This division of labor improves both focus and output
quality.

---

### Step 5: File System Integration

Add persistence for tracking applications:

```python
@tool
def save_application(job_title: str, company: str, cover_letter: str) -> str:
    """Save a generated cover letter to disk."""
    filename = f"output/{company}_{job_title.replace(' ', '_')}.txt"
    with open(filename, "w") as f:
        f.write(cover_letter)
    return f"Saved application to {filename}"
```

File system integration enables true "memory" — the agent can reference past
applications and build knowledge over time, rather than starting fresh with each
session.

---

### Complete Example Output

Here's what the final system produces:

```
$ python main.py

Enter job title: Software Engineer
Enter location: Remote

Current tasks:
- Searching for Software Engineer positions in Remote...
- Found 12 matching positions
- Generating cover letters...

Saved application to output/Google_Software_Engineer.txt
Saved application to output/Meta_Software_Engineer.txt
Saved application to output/Stripe_Software_Engineer.txt

Complete! 3 applications generated.
```
