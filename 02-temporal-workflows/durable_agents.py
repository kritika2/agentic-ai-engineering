"""
Durable Agents with Temporal Workflows
Implements long-running, fault-tolerant agent processes using Temporal.

Key Benefits:
- Survives process crashes and restarts
- Maintains state across failures
- Automatic retry logic with exponential backoff
- Scalable orchestration of multiple agents
- Human-in-the-loop capabilities
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Temporal imports (install with: pip install temporalio)
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    print("⚠️  Temporal not installed. Install with: pip install temporalio")
    TEMPORAL_AVAILABLE = False
    
    # Mock decorators for demo purposes
    class workflow:
        @staticmethod
        def defn(cls):
            return cls
        
        @staticmethod
        def run(func):
            return func
        
        @staticmethod
        def execute_activity(activity_func, *args, **kwargs):
            return activity_func(*args, **kwargs)
    
    class activity:
        @staticmethod
        def defn(func):
            return func

@dataclass
class AgentTask:
    """Represents a task for the agent to complete"""
    id: str
    description: str
    priority: int = 1
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentState:
    """Persistent agent state that survives failures"""
    agent_id: str
    current_task: Optional[AgentTask] = None
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None
    memory: Dict[str, Any] = None
    last_checkpoint: float = 0
    
    def __post_init__(self):
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.failed_tasks is None:
            self.failed_tasks = []
        if self.memory is None:
            self.memory = {}

# Temporal Activities (the actual work units)
@activity.defn
async def analyze_task(task: AgentTask) -> Dict[str, Any]:
    """Analyze a task and determine the approach"""
    print(f"🔍 Analyzing task: {task.description}")
    
    # Simulate analysis time
    await asyncio.sleep(2)
    
    # Determine task complexity and required tools
    complexity = "simple"
    if len(task.description) > 100:
        complexity = "complex"
    elif "search" in task.description.lower() or "research" in task.description.lower():
        complexity = "research"
    
    required_tools = []
    if "calculate" in task.description.lower():
        required_tools.append("calculator")
    if "search" in task.description.lower():
        required_tools.append("search_engine")
    if "email" in task.description.lower():
        required_tools.append("email_client")
    
    return {
        "task_id": task.id,
        "complexity": complexity,
        "required_tools": required_tools,
        "estimated_duration": 30 if complexity == "simple" else 120,
        "analysis_timestamp": time.time()
    }

@activity.defn
async def execute_task_step(task: AgentTask, step_info: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single step of the task"""
    print(f"⚡ Executing step for task: {task.id}")
    
    # Simulate step execution
    duration = step_info.get("estimated_duration", 30)
    await asyncio.sleep(min(duration / 10, 5))  # Scaled down for demo
    
    # Simulate potential failure (10% chance)
    import random
    if random.random() < 0.1:
        raise Exception(f"Simulated failure in step execution for task {task.id}")
    
    return {
        "step_completed": True,
        "result": f"Completed step for {task.description}",
        "timestamp": time.time(),
        "duration": duration
    }

@activity.defn
async def validate_result(task: AgentTask, result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the task execution result"""
    print(f"✅ Validating result for task: {task.id}")
    
    await asyncio.sleep(1)
    
    # Simple validation logic
    is_valid = result.get("step_completed", False)
    
    return {
        "is_valid": is_valid,
        "validation_timestamp": time.time(),
        "quality_score": 0.9 if is_valid else 0.1
    }

@activity.defn
async def update_agent_memory(agent_id: str, task: AgentTask, result: Dict[str, Any]) -> Dict[str, Any]:
    """Update agent's long-term memory with task results"""
    print(f"🧠 Updating memory for agent: {agent_id}")
    
    # In production, this would update a database
    memory_entry = {
        "task_id": task.id,
        "task_description": task.description,
        "result_summary": result.get("result", "No result"),
        "timestamp": time.time(),
        "success": result.get("step_completed", False)
    }
    
    return {
        "memory_updated": True,
        "entry": memory_entry
    }

@activity.defn
async def notify_human(agent_id: str, message: str, requires_approval: bool = False) -> Dict[str, Any]:
    """Notify human operator (simulate human-in-the-loop)"""
    print(f"👤 Human notification for agent {agent_id}: {message}")
    
    if requires_approval:
        print("⏳ Waiting for human approval...")
        # In production, this would wait for actual human input
        await asyncio.sleep(3)  # Simulate human response time
        
        # Simulate approval (80% chance)
        import random
        approved = random.random() < 0.8
        
        return {
            "notification_sent": True,
            "approval_required": True,
            "approved": approved,
            "response_time": 3
        }
    
    return {
        "notification_sent": True,
        "approval_required": False
    }

# Temporal Workflow (the orchestration logic)
@workflow.defn
class DurableAgentWorkflow:
    """
    Durable agent workflow that survives failures and maintains state
    """
    
    def __init__(self):
        self.state = None
        self.retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=3,
            backoff_coefficient=2.0
        ) if TEMPORAL_AVAILABLE else None
    
    @workflow.run
    async def run(self, agent_id: str, tasks: List[AgentTask]) -> Dict[str, Any]:
        """Main workflow execution"""
        
        # Initialize agent state
        self.state = AgentState(agent_id=agent_id)
        
        print(f"🚀 Starting durable agent workflow for {agent_id}")
        print(f"📋 Tasks to process: {len(tasks)}")
        
        results = []
        
        for task in tasks:
            try:
                # Update current task in state
                self.state.current_task = task
                
                # Process the task through multiple steps
                task_result = await self._process_task(task)
                
                # Update state
                if task_result.get("success", False):
                    self.state.completed_tasks.append(task.id)
                else:
                    self.state.failed_tasks.append(task.id)
                
                results.append(task_result)
                
                # Checkpoint state
                self.state.last_checkpoint = time.time()
                
            except Exception as e:
                print(f"❌ Task {task.id} failed: {str(e)}")
                self.state.failed_tasks.append(task.id)
                results.append({
                    "task_id": task.id,
                    "success": False,
                    "error": str(e)
                })
        
        # Final summary
        summary = {
            "agent_id": agent_id,
            "total_tasks": len(tasks),
            "completed": len(self.state.completed_tasks),
            "failed": len(self.state.failed_tasks),
            "results": results,
            "final_state": self.state
        }
        
        # Notify human of completion
        if TEMPORAL_AVAILABLE:
            await workflow.execute_activity(
                notify_human,
                agent_id,
                f"Agent {agent_id} completed {len(self.state.completed_tasks)}/{len(tasks)} tasks",
                start_to_close_timeout=timedelta(seconds=30)
            )
        else:
            await notify_human(agent_id, f"Agent {agent_id} completed {len(self.state.completed_tasks)}/{len(tasks)} tasks")
        
        return summary
    
    async def _process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a single task with full error handling and retries"""
        
        print(f"🎯 Processing task: {task.id}")
        
        try:
            # Step 1: Analyze the task
            if TEMPORAL_AVAILABLE:
                analysis = await workflow.execute_activity(
                    analyze_task,
                    task,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=self.retry_policy
                )
            else:
                analysis = await analyze_task(task)
            
            # Step 2: Check if human approval is needed for complex tasks
            if analysis["complexity"] == "complex":
                if TEMPORAL_AVAILABLE:
                    approval = await workflow.execute_activity(
                        notify_human,
                        self.state.agent_id,
                        f"Complex task requires approval: {task.description}",
                        True,
                        start_to_close_timeout=timedelta(minutes=10)
                    )
                else:
                    approval = await notify_human(self.state.agent_id, f"Complex task requires approval: {task.description}", True)
                
                if not approval.get("approved", False):
                    return {
                        "task_id": task.id,
                        "success": False,
                        "reason": "Human approval denied"
                    }
            
            # Step 3: Execute the task
            if TEMPORAL_AVAILABLE:
                execution_result = await workflow.execute_activity(
                    execute_task_step,
                    task,
                    analysis,
                    start_to_close_timeout=timedelta(seconds=task.timeout_seconds),
                    retry_policy=self.retry_policy
                )
            else:
                execution_result = await execute_task_step(task, analysis)
            
            # Step 4: Validate the result
            if TEMPORAL_AVAILABLE:
                validation = await workflow.execute_activity(
                    validate_result,
                    task,
                    execution_result,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                validation = await validate_result(task, execution_result)
            
            # Step 5: Update agent memory
            if validation["is_valid"]:
                if TEMPORAL_AVAILABLE:
                    memory_update = await workflow.execute_activity(
                        update_agent_memory,
                        self.state.agent_id,
                        task,
                        execution_result,
                        start_to_close_timeout=timedelta(seconds=30)
                    )
                else:
                    memory_update = await update_agent_memory(self.state.agent_id, task, execution_result)
            
            return {
                "task_id": task.id,
                "success": validation["is_valid"],
                "analysis": analysis,
                "execution_result": execution_result,
                "validation": validation,
                "quality_score": validation["quality_score"]
            }
            
        except Exception as e:
            print(f"❌ Error processing task {task.id}: {str(e)}")
            return {
                "task_id": task.id,
                "success": False,
                "error": str(e)
            }

# Workflow execution functions
async def run_durable_agent_demo():
    """Demonstrate durable agent capabilities"""
    
    print("🤖 Durable Agent with Temporal Demo")
    print("=" * 60)
    
    # Create sample tasks
    tasks = [
        AgentTask(
            id="task_1",
            description="Search for the latest AI research papers",
            priority=1
        ),
        AgentTask(
            id="task_2", 
            description="Calculate the ROI for our new product launch with detailed analysis and projections",
            priority=2
        ),
        AgentTask(
            id="task_3",
            description="Send summary email to stakeholders",
            priority=3
        )
    ]
    
    if TEMPORAL_AVAILABLE:
        # Run with actual Temporal (requires Temporal server)
        try:
            client = await Client.connect("localhost:7233")
            
            result = await client.execute_workflow(
                DurableAgentWorkflow.run,
                "agent_001",
                tasks,
                id="durable-agent-demo",
                task_queue="agent-task-queue"
            )
            
            print("\n✅ Workflow completed successfully!")
            print(f"📊 Results: {json.dumps(result, indent=2, default=str)}")
            
        except Exception as e:
            print(f"⚠️  Could not connect to Temporal server: {e}")
            print("Running in simulation mode...")
            await run_simulation_mode(tasks)
    else:
        # Run in simulation mode
        await run_simulation_mode(tasks)

async def run_simulation_mode(tasks: List[AgentTask]):
    """Run the workflow in simulation mode (without Temporal server)"""
    
    print("\n🎭 Running in simulation mode (no Temporal server required)")
    print("-" * 60)
    
    # Create workflow instance
    workflow_instance = DurableAgentWorkflow()
    
    # Execute the workflow
    result = await workflow_instance.run("agent_001", tasks)
    
    print("\n✅ Simulation completed!")
    print(f"📊 Final Results:")
    print(f"   Agent ID: {result['agent_id']}")
    print(f"   Total Tasks: {result['total_tasks']}")
    print(f"   Completed: {result['completed']}")
    print(f"   Failed: {result['failed']}")
    
    print(f"\n📋 Task Details:")
    for task_result in result['results']:
        status = "✅ SUCCESS" if task_result.get('success') else "❌ FAILED"
        print(f"   {task_result['task_id']}: {status}")
        if not task_result.get('success') and 'error' in task_result:
            print(f"      Error: {task_result['error']}")

def setup_temporal_worker():
    """Setup Temporal worker (for production use)"""
    
    if not TEMPORAL_AVAILABLE:
        print("⚠️  Temporal not available. Install with: pip install temporalio")
        return
    
    async def main():
        # Connect to Temporal
        client = await Client.connect("localhost:7233")
        
        # Create worker
        worker = Worker(
            client,
            task_queue="agent-task-queue",
            workflows=[DurableAgentWorkflow],
            activities=[
                analyze_task,
                execute_task_step,
                validate_result,
                update_agent_memory,
                notify_human
            ]
        )
        
        print("🏃 Starting Temporal worker...")
        print("📡 Listening on task queue: agent-task-queue")
        
        # Run worker
        await worker.run()
    
    return main

if __name__ == "__main__":
    print("🚀 Durable Agents with Temporal")
    print("Choose an option:")
    print("1. Run demo (simulation mode)")
    print("2. Start Temporal worker")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_durable_agent_demo())
    elif choice == "2":
        worker_main = setup_temporal_worker()
        if worker_main:
            asyncio.run(worker_main())
    else:
        print("Invalid choice. Running demo...")
        asyncio.run(run_durable_agent_demo())
