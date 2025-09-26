# Agentic AI Engineering

Building intelligent systems that can reason, use tools, and work together. This repo implements concepts from the Agentic AI for Engineers series, focusing on Temporal workflows for reliable agent orchestration.

## What's Inside

This project takes theoretical AI agent concepts and makes them actually work in production:

**Core Agent Stuff**
- Multi-step agents that go beyond simple prompts
- Memory systems, tool integration, planning logic
- Real examples: commerce bots, coding assistants, document processors

**Temporal Integration** (the main thing)
- Long-running processes that don't die when things crash
- State that persists across restarts
- Smart retry logic and error recovery
- Coordinating multiple agents at scale

**Production Ready**
- API integrations, database connections
- Multi-agent collaboration patterns
- Monitoring and safety controls

## Project Structure

```
agentic-ai-engineering/
├── 01-core-agents/                 # Basic agent stuff
│   ├── simple_agent.py            # LLM + memory + tools
│   ├── planner_executor.py        # Planning and execution
│   └── feedback_loops.py          # Self-improvement
├── 02-temporal-workflows/          # Temporal integration (main focus)
│   ├── durable_agents.py          # Long-running processes
│   ├── multi_step_workflows.py    # Complex workflows
│   └── agent_orchestration.py     # Multi-agent coordination
├── 03-tool-integration/            # External tools
│   ├── api_tools.py               # REST APIs
│   ├── database_tools.py          # Database stuff
│   └── search_tools.py            # Search tools
├── 04-multi-agent-systems/        # Agent collaboration
│   ├── crew_ai_patterns.py        # Multi-agent workflows
│   ├── communication.py           # Inter-agent messaging
│   └── role_specialization.py     # Specialized roles
├── examples/                       # Working examples
│   ├── commerce_assistant/         # E-commerce bot
│   ├── coding_assistant/           # Dev workflow agent
│   ├── document_processor/         # Document analysis
│   └── internal_tools_agent/       # Internal tools
└── temporal_setup/                 # Temporal infrastructure
    ├── docker-compose.yml         # Local cluster
    ├── workflows/                 # Workflow definitions
    └── activities/                # Activity definitions
```

## Why Temporal?

Regular AI agents suck because:
- They crash and lose everything
- No memory between steps
- No retry logic when things break
- Can't coordinate multiple agents

Temporal fixes this:
- Workflows survive crashes and restarts
- State persists across failures
- Smart retry logic built-in
- Can coordinate tons of agents

## Getting Started

```bash
git clone https://github.com/kritika2/agentic-ai-engineering.git
cd agentic-ai-engineering

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Temporal (optional)
cd temporal_setup
docker-compose up -d

# Try the examples
python 01-core-agents/simple_agent.py
python 02-temporal-workflows/durable_agents.py
python examples/commerce_assistant/commerce_agent.py
```

## Learning Path

**Start Here:**
1. `simple_agent.py` - Basic LLM + memory + tools
2. `planner_executor.py` - Breaking tasks into steps

**Temporal Stuff (the good part):**
3. `durable_agents.py` - Long-running processes that don't die
4. `multi_step_workflows.py` - Complex workflows with branching

**Production:**
5. `03-tool-integration/` - APIs, databases, search
6. `04-multi-agent-systems/` - Multiple agents working together

## Examples

**Commerce Assistant** - Browses products, compares options, negotiates deals
**Coding Assistant** - Plans code, writes it, runs tests, debugs failures
**Document Processor** - Analyzes contracts, extracts key info

Check the `examples/` folder for working code.

## Tech Stack

**Core:** Python 3.9+, LangChain, Temporal, OpenAI/Anthropic APIs
**Tools:** Requests, SQLAlchemy, BeautifulSoup, Pandas
**Infrastructure:** Docker, Redis, PostgreSQL, FastAPI

## Key Concepts

- **Agent Memory** - Persistent state that survives restarts
- **Tool Use** - External APIs with automatic retries
- **Planning** - Multi-step tasks that can resume anywhere
- **Collaboration** - Multiple agents working together
- **Feedback Loops** - Agents that learn and improve

## Production Notes

**Scaling:** Horizontal scaling with Temporal workers, load balancing
**Reliability:** Auto-retries, circuit breakers, graceful degradation
**Monitoring:** Temporal UI, structured logging, metrics
**Safety:** Human approval gates, rate limiting, content filtering

## Next Steps

1. Try the examples - start simple, work up to Temporal
2. Build your own agent using these patterns
3. Deploy with Temporal for production reliability

## Roadmap

**Done:**
- Core agent architecture
- Temporal integration
- Commerce assistant example

**Coming:**
- More examples (coding assistant, document processor)
- Multi-agent collaboration patterns
- Production monitoring and security

---

Built for engineers who want to build real AI agents that actually work in production.

Based on the "Agentic AI for Engineers" series. Still adding examples and features.
