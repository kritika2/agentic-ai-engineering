"""
Durable agents using Temporal - the main point is they don't die when stuff crashes.
Keeps state, retries failed stuff, can coordinate multiple agents.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Note: this file got pretty long, should probably split it up

# Try to import Temporal, fall back to mocks if not available
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    print("No Temporal - install with: pip install temporalio")
    TEMPORAL_AVAILABLE = False
    
    # Fake decorators so the code still runs
    class workflow:
        @staticmethod
        def defn(cls): return cls
        @staticmethod
        def run(func): return func
        @staticmethod
        def execute_activity(activity_func, *args, **kwargs):
            return activity_func(*args, **kwargs)
    
    class activity:
        @staticmethod
        def defn(func): return func

@dataclass
class AgentTask:
    """A task for the agent"""
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
    """Agent state that persists across crashes"""
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

# Activities - the actual work functions
@activity.defn
async def analyze_task(task: AgentTask) -> Dict[str, Any]:
    """Figure out what this task needs"""
    print(f"Analyzing: {task.description}")
    
    await asyncio.sleep(2)  # fake thinking time
    
    # Basic complexity detection
    complexity = "simple"
    if len(task.description) > 100:
        complexity = "complex"
    elif "search" in task.description.lower() or "research" in task.description.lower():
        complexity = "research"
    
    # What tools do we need?
    tools_needed = []
    desc_lower = task.description.lower()
    if "calculate" in desc_lower:
        tools_needed.append("calculator")
    if "search" in desc_lower:
        tools_needed.append("search_engine")
    if "email" in desc_lower:
        tools_needed.append("email_client")
    
    return {
        "task_id": task.id,
        "complexity": complexity,
        "required_tools": tools_needed,
        "estimated_duration": 30 if complexity == "simple" else 120,
        "analysis_timestamp": time.time()
    }

@activity.defn
async def execute_task_step(task: AgentTask, step_info: Dict[str, Any]) -> Dict[str, Any]:
    """Do the actual work"""
    print(f"Working on: {task.id}")
    
    # Pretend to work
    duration = step_info.get("estimated_duration", 30)
    await asyncio.sleep(min(duration / 10, 5))  # speed up for demo
    
    # Sometimes fail randomly (10% chance)
    import random
    if random.random() < 0.1:
        raise Exception(f"Random failure for {task.id} - this is normal")
    
    return {
        "step_completed": True,
        "result": f"Did the thing for {task.description}",
        "timestamp": time.time(),
        "duration": duration
    }

@activity.defn
async def validate_result(task: AgentTask, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the result is good"""
    print(f"Checking result for: {task.id}")
    
    await asyncio.sleep(1)
    
    # Basic validation
    is_good = result.get("step_completed", False)
    
    return {
        "is_valid": is_good,
        "validation_timestamp": time.time(),
        "quality_score": 0.9 if is_good else 0.1
    }

@activity.defn
async def update_agent_memory(agent_id: str, task: AgentTask, result: Dict[str, Any]) -> Dict[str, Any]:
    """Save stuff to agent memory"""
    print(f"Updating memory for: {agent_id}")
    
    # TODO: save to real database
    memory_entry = {
        "task_id": task.id,
        "task_description": task.description,
        "result_summary": result.get("result", "Nothing happened"),
        "timestamp": time.time(),
        "success": result.get("step_completed", False)
    }
    
    return {
        "memory_updated": True,
        "entry": memory_entry
    }

@activity.defn
async def notify_human(agent_id: str, message: str, requires_approval: bool = False) -> Dict[str, Any]:
    """Tell the human what's up"""
    print(f"Hey human! Agent {agent_id}: {message}")
    
    if requires_approval:
        print("Waiting for approval...")
        await asyncio.sleep(3)  # pretend human is thinking
        
        # Fake approval (80% yes)
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

# Main workflow - this is where the magic happens
@workflow.defn
class DurableAgentWorkflow:
    """Agent workflow that doesn't die when things break"""
    
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
        """Run the agent workflow"""
        
        self.state = AgentState(agent_id=agent_id)
        
        print(f"Starting agent {agent_id}")
        print(f"Got {len(tasks)} tasks to do")
        
        results = []
        
        for task in tasks:
            try:
                self.state.current_task = task
                
                # Do the task
                task_result = await self._process_task(task)
                
                # Track what happened
                if task_result.get("success", False):
                    self.state.completed_tasks.append(task.id)
                else:
                    self.state.failed_tasks.append(task.id)
                
                results.append(task_result)
                self.state.last_checkpoint = time.time()
                
            except Exception as e:
                print(f"Task {task.id} blew up: {str(e)}")
                self.state.failed_tasks.append(task.id)
                results.append({
                    "task_id": task.id,
                    "success": False,
                    "error": str(e)
                })
        
        # Wrap up
        summary = {
            "agent_id": agent_id,
            "total_tasks": len(tasks),
            "completed": len(self.state.completed_tasks),
            "failed": len(self.state.failed_tasks),
            "results": results,
            "final_state": self.state
        }
        
        # Tell the human we're done
        if TEMPORAL_AVAILABLE:
            await workflow.execute_activity(
                notify_human,
                agent_id,
                f"Agent {agent_id} finished: {len(self.state.completed_tasks)}/{len(tasks)} tasks done",
                start_to_close_timeout=timedelta(seconds=30)
            )
        else:
            await notify_human(agent_id, f"Agent {agent_id} finished: {len(self.state.completed_tasks)}/{len(tasks)} tasks done")
        
        return summary
    
    async def _process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Do a single task"""
        
        print(f"Working on: {task.id}")
        
        try:
            # Step 1: Figure out what we need to do
            if TEMPORAL_AVAILABLE:
                analysis = await workflow.execute_activity(
                    analyze_task, task,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=self.retry_policy
                )
            else:
                analysis = await analyze_task(task)
            
            # Step 2: Ask human if it's complex
            if analysis["complexity"] == "complex":
                if TEMPORAL_AVAILABLE:
                    approval = await workflow.execute_activity(
                        notify_human, self.state.agent_id,
                        f"Complex task needs approval: {task.description}", True,
                        start_to_close_timeout=timedelta(minutes=10)
                    )
                else:
                    approval = await notify_human(self.state.agent_id, f"Complex task needs approval: {task.description}", True)
                
                if not approval.get("approved", False):
                    return {"task_id": task.id, "success": False, "reason": "Human said no"}
            
            # Step 3: Do the work
            if TEMPORAL_AVAILABLE:
                execution_result = await workflow.execute_activity(
                    execute_task_step, task, analysis,
                    start_to_close_timeout=timedelta(seconds=task.timeout_seconds),
                    retry_policy=self.retry_policy
                )
            else:
                execution_result = await execute_task_step(task, analysis)
            
            # Step 4: Check if it's good
            if TEMPORAL_AVAILABLE:
                validation = await workflow.execute_activity(
                    validate_result, task, execution_result,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                validation = await validate_result(task, execution_result)
            
            # Step 5: Save to memory if it worked
            if validation["is_valid"]:
                if TEMPORAL_AVAILABLE:
                    await workflow.execute_activity(
                        update_agent_memory, self.state.agent_id, task, execution_result,
                        start_to_close_timeout=timedelta(seconds=30)
                    )
                else:
                    await update_agent_memory(self.state.agent_id, task, execution_result)
            
            return {
                "task_id": task.id,
                "success": validation["is_valid"],
                "analysis": analysis,
                "execution_result": execution_result,
                "validation": validation,
                "quality_score": validation["quality_score"]
            }
            
        except Exception as e:
            print(f"Task {task.id} failed: {str(e)}")
            return {"task_id": task.id, "success": False, "error": str(e)}

# Demo functions
async def run_durable_agent_demo():
    """Try out the durable agent"""
    
    print("Durable Agent Demo")
    print("=" * 50)
    
    # Some test tasks
    tasks = [
        AgentTask("task_1", "Search for AI research papers", priority=1),
        AgentTask("task_2", "Calculate ROI for new product launch with detailed analysis and projections", priority=2),
        AgentTask("task_3", "Send summary email to stakeholders", priority=3)
    ]
    
    if TEMPORAL_AVAILABLE:
        # Try to use real Temporal
        try:
            client = await Client.connect("localhost:7233")
            result = await client.execute_workflow(
                DurableAgentWorkflow.run, "agent_001", tasks,
                id="durable-agent-demo", task_queue="agent-task-queue"
            )
            print("\nTemporal workflow done!")
            print(f"Results: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            print(f"No Temporal server: {e}")
            print("Running fake mode...")
            await run_simulation_mode(tasks)
    else:
        await run_simulation_mode(tasks)

async def run_simulation_mode(tasks: List[AgentTask]):
    """Run without Temporal server"""
    
    print("\nSimulation mode (no Temporal needed)")
    print("-" * 50)
    
    workflow_instance = DurableAgentWorkflow()
    result = await workflow_instance.run("agent_001", tasks)
    
    print("\nDone!")
    print(f"Agent: {result['agent_id']}")
    print(f"Tasks: {result['total_tasks']} total, {result['completed']} done, {result['failed']} failed")
    
    print(f"\nDetails:")
    for task_result in result['results']:
        status = "OK" if task_result.get('success') else "FAIL"
        print(f"   {task_result['task_id']}: {status}")
        if not task_result.get('success') and 'error' in task_result:
            print(f"      Error: {task_result['error']}")

def setup_temporal_worker():
    """Setup worker for production"""
    
    if not TEMPORAL_AVAILABLE:
        print("Need Temporal: pip install temporalio")
        return
    
    async def main():
        client = await Client.connect("localhost:7233")
        worker = Worker(
            client, task_queue="agent-task-queue",
            workflows=[DurableAgentWorkflow],
            activities=[analyze_task, execute_task_step, validate_result, update_agent_memory, notify_human]
        )
        
        print("Starting worker...")
        print("Queue: agent-task-queue")
        await worker.run()
    
    return main

if __name__ == "__main__":
    print("Durable Agents")
    print("1. Run demo")
    print("2. Start worker")
    
    choice = input("Pick (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_durable_agent_demo())
    elif choice == "2":
        worker_main = setup_temporal_worker()
        if worker_main:
            asyncio.run(worker_main())
    else:
        print("Running demo...")
        asyncio.run(run_durable_agent_demo())

