"""
Multi-step workflows using Temporal.
Built this after realizing simple agents aren't enough for complex stuff.
Handles approvals, retries, and all that enterprise nonsense.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Temporal imports with fallback
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    # Mock classes
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

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class WorkflowRequest:
    """A request to start a workflow"""
    id: str
    type: str
    description: str
    requester: str
    priority: Priority = Priority.NORMAL
    parameters: Dict[str, Any] = None
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ApprovalRequest:
    """Request for human approval"""
    id: str
    workflow_id: str
    title: str
    description: str
    options: List[str]
    required_role: str = "manager"
    timeout_minutes: int = 60
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 0
    execution_time: float = 0
    approvals_requested: int = 0

# Activities for different workflow types

@activity.defn
async def validate_request(request: WorkflowRequest) -> Dict[str, Any]:
    """Validate incoming workflow request"""
    print(f"Validating request: {request.id}")
    
    await asyncio.sleep(1)
    
    # basic validation - nothing fancy
    issues = []
    
    if not request.description.strip():
        issues.append("Description is required")
    
    if request.priority == Priority.URGENT and not request.deadline:
        issues.append("Urgent requests must have a deadline")
    
    # check if we actually support this workflow type
    supported_types = ["data_processing", "content_creation", "approval_workflow", "research_task"]
    if request.type not in supported_types:
        issues.append(f"Unsupported workflow type: {request.type}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "estimated_duration": 30 if request.priority == Priority.LOW else 15,
        "requires_approval": request.priority in [Priority.HIGH, Priority.URGENT]
    }

@activity.defn
async def process_data(data_params: Dict[str, Any]) -> Dict[str, Any]:
    """Process data according to parameters"""
    print(f"Processing data: {data_params.get('source', 'unknown')}")
    
    await asyncio.sleep(3)  # Simulate processing time
    
    # Mock data processing
    source = data_params.get("source", "default")
    operation = data_params.get("operation", "analyze")
    
    # simulate processing - sometimes fails to test error handling
    import random
    success = random.random() > 0.1  # 90% success rate
    
    if success:
        return {
            "success": True,
            "source": source,
            "operation": operation,
            "records_processed": random.randint(100, 1000),
            "output_location": f"/data/processed/{source}_{int(time.time())}.json",
            "processing_time": 3.0
        }
    else:
        return {
            "success": False,
            "error": "Data processing failed - corrupt input data",
            "source": source
        }

@activity.defn
async def create_content(content_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create content based on parameters"""
    print(f"Creating content: {content_params.get('type', 'unknown')}")
    
    await asyncio.sleep(4)  # Content creation takes time
    
    content_type = content_params.get("type", "article")
    topic = content_params.get("topic", "general")
    length = content_params.get("length", "medium")
    
    # Mock content creation
    word_counts = {"short": 300, "medium": 800, "long": 1500}
    word_count = word_counts.get(length, 800)
    
    return {
        "success": True,
        "content_type": content_type,
        "topic": topic,
        "word_count": word_count,
        "content": f"Generated {content_type} about {topic} ({word_count} words)",
        "quality_score": 0.85,
        "creation_time": 4.0
    }

@activity.defn
async def conduct_research(research_params: Dict[str, Any]) -> Dict[str, Any]:
    """Conduct research on a topic"""
    print(f"Researching: {research_params.get('topic', 'unknown')}")
    
    await asyncio.sleep(5)  # Research takes time
    
    topic = research_params.get("topic", "general")
    depth = research_params.get("depth", "basic")
    sources = research_params.get("sources", ["web", "academic"])
    
    # Mock research results
    findings = {
        "topic": topic,
        "depth": depth,
        "sources_consulted": len(sources) * 5,
        "key_findings": [
            f"Key insight 1 about {topic}",
            f"Important trend in {topic}",
            f"Future implications for {topic}"
        ],
        "confidence_score": 0.8,
        "research_time": 5.0
    }
    
    return {"success": True, "findings": findings}

@activity.defn
async def request_approval(approval: ApprovalRequest) -> Dict[str, Any]:
    """Request human approval"""
    print(f"Requesting approval: {approval.title}")
    
    # Simulate human decision time
    await asyncio.sleep(min(approval.timeout_minutes * 0.1, 10))  # Speed up for demo
    
    # fake approval decision for testing - usually approves
    import random
    approved = random.random() < 0.8
    
    response = {
        "approval_id": approval.id,
        "approved": approved,
        "approver": "manager@company.com",
        "decision_time": time.time(),
        "comments": "Approved for processing" if approved else "Needs more information"
    }
    
    print(f"Approval {'granted' if approved else 'denied'}: {approval.title}")
    return response

@activity.defn
async def send_notification(recipient: str, message: str, priority: Priority) -> Dict[str, Any]:
    """Send notification to user"""
    print(f"Notifying {recipient}: {message}")
    
    await asyncio.sleep(0.5)
    
    return {
        "sent": True,
        "recipient": recipient,
        "message": message,
        "priority": priority.name,
        "timestamp": time.time()
    }

@activity.defn
async def cleanup_resources(workflow_id: str) -> Dict[str, Any]:
    """Clean up any temporary resources"""
    print(f"Cleaning up resources for: {workflow_id}")
    
    await asyncio.sleep(1)
    
    return {
        "cleaned": True,
        "workflow_id": workflow_id,
        "resources_cleaned": ["temp_files", "cache_entries", "locks"],
        "cleanup_time": 1.0
    }

# Main multi-step workflow
@workflow.defn
class MultiStepWorkflow:
    """Complex workflow with conditional branching and human approval"""
    
    def __init__(self):
        self.retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=3,
            backoff_coefficient=2.0
        ) if TEMPORAL_AVAILABLE else None
        
        self.steps_completed = 0
        self.total_steps = 0
        self.approvals_requested = 0
    
    @workflow.run
    async def run(self, request: WorkflowRequest) -> WorkflowResult:
        """Execute the multi-step workflow"""
        
        print(f"Starting workflow: {request.id}")
        print(f"Type: {request.type}")
        print(f"Priority: {request.priority.name}")
        
        start_time = time.time()
        
        try:
            # Step 1: Validate the request
            if TEMPORAL_AVAILABLE:
                validation = await workflow.execute_activity(
                    validate_request, request,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                validation = await validate_request(request)
            
            self.steps_completed += 1
            
            if not validation["valid"]:
                return WorkflowResult(
                    workflow_id=request.id,
                    status=WorkflowStatus.FAILED,
                    error=f"Validation failed: {', '.join(validation['issues'])}",
                    steps_completed=self.steps_completed,
                    execution_time=time.time() - start_time
                )
            
            # Step 2: Request approval if needed
            if validation["requires_approval"]:
                approval_request = ApprovalRequest(
                    id=f"approval_{request.id}",
                    workflow_id=request.id,
                    title=f"Approve {request.type} workflow",
                    description=f"Request to execute {request.description}",
                    options=["approve", "deny", "request_changes"],
                    timeout_minutes=30 if request.priority == Priority.URGENT else 60
                )
                
                if TEMPORAL_AVAILABLE:
                    approval_result = await workflow.execute_activity(
                        request_approval, approval_request,
                        start_to_close_timeout=timedelta(minutes=approval_request.timeout_minutes)
                    )
                else:
                    approval_result = await request_approval(approval_request)
                
                self.approvals_requested += 1
                self.steps_completed += 1
                
                if not approval_result["approved"]:
                    return WorkflowResult(
                        workflow_id=request.id,
                        status=WorkflowStatus.CANCELLED,
                        error="Workflow not approved",
                        steps_completed=self.steps_completed,
                        approvals_requested=self.approvals_requested,
                        execution_time=time.time() - start_time
                    )
            
            # Step 3: Execute workflow based on type
            workflow_result = await self._execute_workflow_type(request)
            self.steps_completed += 1
            
            if not workflow_result.get("success", False):
                return WorkflowResult(
                    workflow_id=request.id,
                    status=WorkflowStatus.FAILED,
                    error=workflow_result.get("error", "Workflow execution failed"),
                    steps_completed=self.steps_completed,
                    approvals_requested=self.approvals_requested,
                    execution_time=time.time() - start_time
                )
            
            # Step 4: Send completion notification
            if TEMPORAL_AVAILABLE:
                await workflow.execute_activity(
                    send_notification,
                    request.requester,
                    f"Workflow {request.id} completed successfully",
                    request.priority,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                await send_notification(
                    request.requester,
                    f"Workflow {request.id} completed successfully",
                    request.priority
                )
            
            self.steps_completed += 1
            
            # Step 5: Cleanup
            if TEMPORAL_AVAILABLE:
                await workflow.execute_activity(
                    cleanup_resources, request.id,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                await cleanup_resources(request.id)
            
            self.steps_completed += 1
            
            return WorkflowResult(
                workflow_id=request.id,
                status=WorkflowStatus.COMPLETED,
                result=workflow_result,
                steps_completed=self.steps_completed,
                total_steps=5,
                approvals_requested=self.approvals_requested,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"Workflow {request.id} failed: {str(e)}")
            return WorkflowResult(
                workflow_id=request.id,
                status=WorkflowStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed,
                approvals_requested=self.approvals_requested,
                execution_time=time.time() - start_time
            )
    
    async def _execute_workflow_type(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Execute the specific workflow type"""
        
        if request.type == "data_processing":
            if TEMPORAL_AVAILABLE:
                return await workflow.execute_activity(
                    process_data, request.parameters,
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=self.retry_policy
                )
            else:
                return await process_data(request.parameters)
        
        elif request.type == "content_creation":
            if TEMPORAL_AVAILABLE:
                return await workflow.execute_activity(
                    create_content, request.parameters,
                    start_to_close_timeout=timedelta(minutes=15),
                    retry_policy=self.retry_policy
                )
            else:
                return await create_content(request.parameters)
        
        elif request.type == "research_task":
            if TEMPORAL_AVAILABLE:
                return await workflow.execute_activity(
                    conduct_research, request.parameters,
                    start_to_close_timeout=timedelta(minutes=20),
                    retry_policy=self.retry_policy
                )
            else:
                return await conduct_research(request.parameters)
        
        elif request.type == "approval_workflow":
            # This is a meta-workflow that just handles approvals
            return {"success": True, "message": "Approval workflow completed"}
        
        else:
            return {"success": False, "error": f"Unknown workflow type: {request.type}"}

# Demo function
async def demo_multi_step_workflows():
    """Demo the multi-step workflow system"""
    
    print("Multi-Step Workflows Demo")
    print("=" * 60)
    
    # Create test workflow requests
    test_requests = [
        WorkflowRequest(
            id="wf_001",
            type="data_processing",
            description="Process customer data for monthly report",
            requester="analyst@company.com",
            priority=Priority.NORMAL,
            parameters={
                "source": "customer_database",
                "operation": "aggregate",
                "output_format": "json"
            }
        ),
        WorkflowRequest(
            id="wf_002", 
            type="content_creation",
            description="Create blog post about AI trends",
            requester="marketing@company.com",
            priority=Priority.HIGH,
            parameters={
                "type": "blog_post",
                "topic": "AI trends 2024",
                "length": "long",
                "target_audience": "technical"
            },
            deadline=datetime.now() + timedelta(hours=24)
        ),
        WorkflowRequest(
            id="wf_003",
            type="research_task", 
            description="Research competitive landscape",
            requester="strategy@company.com",
            priority=Priority.URGENT,
            parameters={
                "topic": "AI automation tools",
                "depth": "comprehensive",
                "sources": ["web", "academic", "industry_reports"]
            },
            deadline=datetime.now() + timedelta(hours=12)
        )
    ]
    
    # Execute workflows
    for i, request in enumerate(test_requests, 1):
        print(f"\n{'='*60}")
        print(f"WORKFLOW {i}/{len(test_requests)}")
        print(f"ID: {request.id}")
        print(f"Type: {request.type}")
        print(f"Priority: {request.priority.name}")
        print(f"Description: {request.description}")
        print("-" * 40)
        
        workflow_instance = MultiStepWorkflow()
        result = await workflow_instance.run(request)
        
        print(f"\nRESULT:")
        print(f"Status: {result.status.name}")
        print(f"Steps: {result.steps_completed}/{result.total_steps or 'unknown'}")
        print(f"Time: {result.execution_time:.2f}s")
        
        if result.approvals_requested > 0:
            print(f"Approvals: {result.approvals_requested}")
        
        if result.status == WorkflowStatus.COMPLETED:
            print("Workflow completed successfully")
            if result.result:
                print(f"Result summary: {list(result.result.keys())}")
        elif result.status == WorkflowStatus.FAILED:
            print(f"Workflow failed: {result.error}")
        elif result.status == WorkflowStatus.CANCELLED:
            print(f"Workflow cancelled: {result.error}")
        
        if i < len(test_requests):
            print(f"\nContinuing to next workflow...")
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"Workflows executed: {len(test_requests)}")

# Production setup
def setup_workflow_worker():
    """Setup Temporal worker for production"""
    
    if not TEMPORAL_AVAILABLE:
        print("Temporal not available. Install with: pip install temporalio")
        return None
    
    async def main():
        client = await Client.connect("localhost:7233")
        worker = Worker(
            client,
            task_queue="multi-step-workflow-queue",
            workflows=[MultiStepWorkflow],
            activities=[
                validate_request, process_data, create_content,
                conduct_research, request_approval, send_notification,
                cleanup_resources
            ]
        )
        
        print("Starting multi-step workflow worker...")
        print("Task queue: multi-step-workflow-queue")
        await worker.run()
    
    return main

if __name__ == "__main__":
    print("Multi-Step Workflows")
    print("1. Run demo")
    print("2. Start worker")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_multi_step_workflows())
    elif choice == "2":
        worker_main = setup_workflow_worker()
        if worker_main:
            asyncio.run(worker_main())
        else:
            print("Cannot start worker - Temporal not available")
    else:
        print("Running demo...")
        asyncio.run(demo_multi_step_workflows())
