
## Quick Start

### 1. **Setup Environment**
```bash
git clone https://github.com/kritika2/agentic-ai-engineering.git
cd agentic-ai-engineering

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Start Temporal (Optional)**
```bash
cd temporal_setup
docker-compose up -d

# Verify Temporal is running
# Web UI available at: http://localhost:8080
```

### 3. **Run Basic Agent**
```bash
python 01-core-agents/simple_agent.py
```

### 4. **Try Temporal Workflow**
```bash
python 02-temporal-workflows/durable_agents.py
```

### 5. **Test Commerce Assistant**
```bash
python examples/commerce_assistant/commerce_agent.py
```

## Learning Path

### **Level 1: Agent Fundamentals**
1. **Simple Agent** (`01-core-agents/simple_agent.py`)
   - LLM + basic memory + tool use
   - Single-turn interactions

2. **Planner-Executor Pattern** (`01-core-agents/planner_executor.py`)
   - Break tasks into steps
   - Execute and validate results

### **Level 2: Temporal Integration**
3. **Durable Agents** (`02-temporal-workflows/durable_agents.py`)
   - Long-running processes
   - Failure recovery
   - State persistence

4. **Multi-Step Workflows** (`02-temporal-workflows/multi_step_workflows.py`)
   - Complex task orchestration
   - Conditional branching
   - Human-in-the-loop

### **Level 3: Production Systems**
5. **Tool Integration** (`03-tool-integration/`)
   - External API calls
   - Database operations
   - Search and retrieval

6. **Multi-Agent Systems** (`04-multi-agent-systems/`)
   - Agent collaboration
   - Role specialization
   - Communication patterns

## Real-World Examples

### Commerce Assistant
```python
# Autonomous e-commerce agent with Temporal
@workflow.defn
class CommerceAgentWorkflow:
    async def run(self, user_request: str) -> str:
        # Browse products
        products = await workflow.execute_activity(
            browse_products, user_request
        )
        
        # Compare options
        comparison = await workflow.execute_activity(
            compare_products, products
        )
        
        # Negotiate deals (if applicable)
        if comparison.needs_negotiation:
            deal = await workflow.execute_activity(
                negotiate_deal, comparison.best_option
            )
        
        return deal.final_recommendation
```

### Coding Assistant
```python
# Multi-step development workflow
@workflow.defn
class CodingAssistantWorkflow:
    async def run(self, task: str) -> CodeResult:
        # Plan implementation
        plan = await workflow.execute_activity(plan_code, task)
        
        # Write code
        code = await workflow.execute_activity(write_code, plan)
        
        # Run tests
        test_results = await workflow.execute_activity(run_tests, code)
        
        # Debug if needed
        if not test_results.passed:
            code = await workflow.execute_activity(debug_code, code, test_results)
        
        return CodeResult(code=code, tests=test_results)
```

## Technologies Used

### **Core Stack**
- **Python 3.9+** - Primary language
- **LangChain** - LLM orchestration and tool integration
- **Temporal** - Durable workflow orchestration
- **OpenAI/Anthropic APIs** - Language models

### **Agent Tools**
- **Requests** - HTTP API integration
- **SQLAlchemy** - Database operations
- **BeautifulSoup** - Web scraping
- **Pandas** - Data processing

### **Infrastructure**
- **Docker** - Temporal cluster setup
- **Redis** - Agent state caching
- **PostgreSQL** - Persistent storage
- **FastAPI** - Agent API endpoints

## Key Concepts Implemented

| Concept | Implementation | Temporal Benefit |
|---------|---------------|------------------|
| **Agent Memory** | Persistent state across steps | Survives process restarts |
| **Tool Use** | External API integration | Automatic retry on failures |
| **Planning** | Multi-step task breakdown | Resumable from any step |
| **Collaboration** | Multi-agent workflows | Coordinated execution |
| **Feedback Loops** | Self-improvement cycles | Long-term learning persistence |

## Production Considerations

### **Scalability**
- **Horizontal scaling** with Temporal workers
- **Load balancing** across agent instances
- **Resource management** for long-running processes

### **Reliability**
- **Automatic retries** with exponential backoff
- **Circuit breakers** for external services
- **Graceful degradation** when tools fail

### **Observability**
- **Temporal UI** for workflow monitoring
- **Structured logging** for agent decisions
- **Metrics collection** for performance tracking

### **Safety**
- **Human approval** for sensitive operations
- **Rate limiting** for external API calls
- **Content filtering** for agent outputs

## Next Steps

1. **Explore the examples** - Start with simple agents, progress to Temporal workflows
2. **Build your own agent** - Use the patterns for your specific use case
3. **Deploy to production** - Leverage Temporal for reliable agent orchestration
4. **Join the community** - Share your agent implementations and learnings

## Development Roadmap

### **Phase 1: Foundation**
- Core agent architecture
- Temporal workflow integration  
- Commerce assistant example
- Production infrastructure setup

### **Phase 2: Advanced Examples** (Coming Soon)
- **Coding Assistant** - Multi-step development workflows
- **Document Processor** - Contract analysis and summarization  
- **Internal Tools Agent** - Dashboard and ticket management

### **Phase 3: Multi-Agent Systems** (Planned)
- **Agent Collaboration** - CrewAI integration patterns
- **Role Specialization** - Specialized agent types
- **Communication Protocols** - Inter-agent messaging

### **Phase 4: Production Features** (Future)
- **Advanced Monitoring** - Agent performance dashboards
- **Security & Governance** - Access control and audit trails
- **Scaling Patterns** - High-throughput agent orchestration

---

**Built for the future of intelligent automation**

*This repository implements concepts from the "Agentic AI for Engineers" series. Each example includes detailed documentation and is production-ready.*

**Note**: This is an active project! Follow for updates as we add more examples and advanced features...
