"""
Agent that actually plans stuff before doing it.
Got tired of the simple agent being too random, so built this one
to break tasks down properly.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskStep:
    """A single step in a plan"""
    id: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class ExecutionPlan:
    """A complete execution plan"""
    id: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    
    def get_ready_steps(self) -> List[TaskStep]:
        """Get steps that are ready to execute"""
        ready = []
        for step in self.steps:
            if step.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_done = all(
                any(s.id == dep_id and s.status == TaskStatus.COMPLETED 
                    for s in self.steps)
                for dep_id in step.dependencies
            )
            
            if deps_done:
                ready.append(step)
        
        return ready
    
    def is_complete(self) -> bool:
        """Check if all steps are done"""
        return all(step.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] 
                  for step in self.steps)
    
    def has_failures(self) -> bool:
        """Check if any steps failed"""
        return any(step.status == TaskStatus.FAILED for step in self.steps)

class LLMPlanner:
    """Plans tasks using LLM reasoning"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
    
    def create_plan(self, task_description: str, available_tools: List[str]) -> ExecutionPlan:
        """Break down a task into steps"""
        print(f"Planning: {task_description}")
        
        # mock planning for now - will make this smarter later
        plan_id = f"plan_{int(time.time())}"
        plan = ExecutionPlan(id=plan_id, description=task_description)
        
        # basic pattern matching - should use proper NLP eventually
        task_lower = task_description.lower()
        
        if "research" in task_lower or "find" in task_lower:
            plan.steps = [
                TaskStep("step_1", "Search for information", "search", 
                        {"query": task_description}),
                TaskStep("step_2", "Analyze search results", "analyze", 
                        {"data_source": "search_results"}, dependencies=["step_1"]),
                TaskStep("step_3", "Summarize findings", "summarize", 
                        {"content": "analysis_results"}, dependencies=["step_2"])
            ]
        
        elif "calculate" in task_lower or "compute" in task_lower:
            plan.steps = [
                TaskStep("step_1", "Extract numbers and operations", "parse", 
                        {"text": task_description}),
                TaskStep("step_2", "Perform calculations", "calculate", 
                        {"expression": "parsed_expression"}, dependencies=["step_1"]),
                TaskStep("step_3", "Validate result", "validate", 
                        {"result": "calculation_result"}, dependencies=["step_2"])
            ]
        
        elif "write" in task_lower or "create" in task_lower:
            plan.steps = [
                TaskStep("step_1", "Gather requirements", "analyze", 
                        {"requirements": task_description}),
                TaskStep("step_2", "Create outline", "outline", 
                        {"topic": "requirements"}, dependencies=["step_1"]),
                TaskStep("step_3", "Write content", "write", 
                        {"outline": "content_outline"}, dependencies=["step_2"]),
                TaskStep("step_4", "Review and edit", "review", 
                        {"content": "written_content"}, dependencies=["step_3"])
            ]
        
        else:
            # fallback plan when I don't know what to do
            plan.steps = [
                TaskStep("step_1", "Understand the task", "analyze", 
                        {"task": task_description}),
                TaskStep("step_2", "Execute main action", "execute", 
                        {"action": "main_task"}, dependencies=["step_1"]),
                TaskStep("step_3", "Verify completion", "validate", 
                        {"result": "execution_result"}, dependencies=["step_2"])
            ]
        
        print(f"Created plan with {len(plan.steps)} steps")
        return plan

class TaskExecutor:
    """Executes individual task steps"""
    
    def __init__(self):
        self.tools = {
            "search": self._search_tool,
            "calculate": self._calculate_tool,
            "analyze": self._analyze_tool,
            "summarize": self._summarize_tool,
            "parse": self._parse_tool,
            "outline": self._outline_tool,
            "write": self._write_tool,
            "review": self._review_tool,
            "validate": self._validate_tool,
            "execute": self._execute_tool
        }
        self.context = {}  # Store results between steps
    
    def execute_step(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        print(f"Executing: {step.description}")
        
        step.start_time = time.time()
        step.status = TaskStatus.IN_PROGRESS
        
        try:
            if step.action in self.tools:
                result = self.tools[step.action](step, context)
                step.result = result
                step.status = TaskStatus.COMPLETED
                
                # Store result in context for next steps
                context[f"{step.id}_result"] = result
                
                print(f"Completed: {step.description}")
                return result
            else:
                raise Exception(f"Unknown action: {step.action}")
        
        except Exception as e:
            step.error = str(e)
            step.status = TaskStatus.FAILED
            print(f"Failed: {step.description} - {str(e)}")
            raise
        
        finally:
            step.end_time = time.time()
    
    def _search_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fake search - will connect to real search API later"""
        query = step.parameters.get("query", "")
        
        # mock search results for testing
        results = [
            {"title": f"Result 1 for {query}", "snippet": "Some relevant information..."},
            {"title": f"Result 2 for {query}", "snippet": "More details about the topic..."},
            {"title": f"Result 3 for {query}", "snippet": "Additional insights..."}
        ]
        
        return {"query": query, "results": results, "count": len(results)}
    
    def _calculate_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock calculator"""
        expression = step.parameters.get("expression", "1+1")
        
        # Get from previous step if needed
        if expression in context:
            expression = context[expression]
        
        try:
            # using eval for now - I know it's bad, will fix later
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in str(expression)):
                result = eval(str(expression))
                return {"expression": expression, "result": result, "success": True}
            else:
                return {"expression": expression, "error": "Invalid expression", "success": False}
        except Exception as e:
            return {"expression": expression, "error": str(e), "success": False}
    
    def _analyze_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock analysis tool"""
        data = step.parameters.get("task", step.parameters.get("requirements", ""))
        
        analysis = {
            "complexity": "medium" if len(str(data)) > 50 else "simple",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "recommendations": ["Do this", "Then that", "Finally this"],
            "confidence": 0.85
        }
        
        return analysis
    
    def _summarize_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock summarization"""
        content = step.parameters.get("content", "")
        
        # Look for content in context
        if content in context:
            content = context[content]
        
        summary = {
            "original_length": len(str(content)),
            "summary": "This is a summary of the key findings and important points.",
            "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
            "compression_ratio": 0.3
        }
        
        return summary
    
    def _parse_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse text to extract useful stuff"""
        text = step.parameters.get("text", "")
        
        # extract numbers and operations - pretty basic regex
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        operators = re.findall(r'[+\-*/]', text)
        
        if numbers and operators:
            expression = numbers[0]
            for i, op in enumerate(operators):
                if i + 1 < len(numbers):
                    expression += f" {op} {numbers[i + 1]}"
        else:
            expression = "1 + 1"  # fallback when nothing found
        
        return {"original_text": text, "parsed_expression": expression, "numbers": numbers}
    
    def _outline_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock outline creation"""
        topic = step.parameters.get("topic", "General Topic")
        
        outline = {
            "title": f"Outline for {topic}",
            "sections": [
                {"title": "Introduction", "points": ["Background", "Objectives"]},
                {"title": "Main Content", "points": ["Key Point 1", "Key Point 2", "Key Point 3"]},
                {"title": "Conclusion", "points": ["Summary", "Next Steps"]}
            ]
        }
        
        return outline
    
    def _write_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock writing tool"""
        outline = step.parameters.get("outline", {})
        
        content = {
            "title": "Generated Content",
            "word_count": 250,
            "content": "This is the generated content based on the outline. It includes all the key points and follows the structure provided.",
            "sections_completed": 3
        }
        
        return content
    
    def _review_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock review tool"""
        content = step.parameters.get("content", "")
        
        review = {
            "grammar_score": 0.92,
            "clarity_score": 0.88,
            "completeness_score": 0.85,
            "suggestions": ["Fix typo in line 3", "Clarify point about X", "Add more examples"],
            "overall_score": 0.88
        }
        
        return review
    
    def _validate_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock validation tool"""
        result = step.parameters.get("result", "")
        
        validation = {
            "is_valid": True,
            "confidence": 0.9,
            "checks_passed": ["Format check", "Content check", "Logic check"],
            "issues_found": []
        }
        
        return validation
    
    def _execute_tool(self, step: TaskStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock generic execution"""
        action = step.parameters.get("action", "generic_task")
        
        return {
            "action": action,
            "status": "completed",
            "output": f"Successfully executed {action}",
            "duration": 2.5
        }

class PlannerExecutorAgent:
    """Agent that plans and executes complex tasks"""
    
    def __init__(self, name: str = "PlannerExecutor"):
        self.name = name
        self.planner = LLMPlanner()
        self.executor = TaskExecutor()
        self.execution_history = []
    
    def process_task(self, task_description: str) -> Dict[str, Any]:
        """Main method: plan and execute a task"""
        
        print(f"\n{self.name}: Processing task")
        print(f"Task: {task_description}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Create execution plan
            available_tools = list(self.executor.tools.keys())
            plan = self.planner.create_plan(task_description, available_tools)
            
            print(f"\nPlan created with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
                print(f"  {i}. {step.description}{deps}")
            
            # Step 2: Execute the plan
            print(f"\nExecuting plan...")
            print("-" * 40)
            
            context = {}
            max_iterations = 20  # Prevent infinite loops
            iteration = 0
            
            while not plan.is_complete() and not plan.has_failures() and iteration < max_iterations:
                ready_steps = plan.get_ready_steps()
                
                if not ready_steps:
                    if not plan.is_complete():
                        print("No ready steps but plan not complete - possible dependency issue")
                        break
                    else:
                        break
                
                # Execute ready steps (could be parallel in real implementation)
                for step in ready_steps:
                    try:
                        self.executor.execute_step(step, context)
                    except Exception as e:
                        print(f"Step failed: {step.id} - {str(e)}")
                        # Continue with other steps if possible
                
                iteration += 1
            
            # Step 3: Analyze results
            end_time = time.time()
            
            completed_steps = [s for s in plan.steps if s.status == TaskStatus.COMPLETED]
            failed_steps = [s for s in plan.steps if s.status == TaskStatus.FAILED]
            
            result = {
                "task": task_description,
                "plan_id": plan.id,
                "success": plan.is_complete() and not plan.has_failures(),
                "total_steps": len(plan.steps),
                "completed_steps": len(completed_steps),
                "failed_steps": len(failed_steps),
                "execution_time": end_time - start_time,
                "plan": plan,
                "context": context
            }
            
            # Store in history
            self.execution_history.append(result)
            
            # Print summary
            print("-" * 40)
            if result["success"]:
                print(f"Task completed successfully!")
                print(f"  Steps: {result['completed_steps']}/{result['total_steps']}")
                print(f"  Time: {result['execution_time']:.2f}s")
            else:
                print(f"Task failed or incomplete")
                print(f"  Completed: {result['completed_steps']}/{result['total_steps']}")
                print(f"  Failed: {result['failed_steps']}")
                if failed_steps:
                    print("  Failures:")
                    for step in failed_steps:
                        print(f"    - {step.description}: {step.error}")
            
            return result
            
        except Exception as e:
            error_result = {
                "task": task_description,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            print(f"Task processing failed: {str(e)}")
            return error_result

def demo_planner_executor():
    """Demo the planner-executor agent"""
    
    print("Planner-Executor Agent Demo")
    print("=" * 70)
    
    agent = PlannerExecutorAgent("PlanBot")
    
    # Test tasks of different types
    test_tasks = [
        "Research the latest trends in artificial intelligence",
        "Calculate the compound interest on $10,000 at 5% for 10 years",
        "Write a brief summary of machine learning applications in healthcare",
        "Find information about renewable energy and create a report"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_tasks)}")
        
        result = agent.process_task(task)
        
        if i < len(test_tasks):
            print(f"\nContinuing to next test...")
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"Total tasks processed: {len(agent.execution_history)}")
    print(f"Successful: {sum(1 for r in agent.execution_history if r['success'])}")
    print(f"Failed: {sum(1 for r in agent.execution_history if not r['success'])}")

if __name__ == "__main__":
    demo_planner_executor()
