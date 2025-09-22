"""
Simple Agent: Basic Building Blocks
Implements the core agent pattern from the article:
- LLM for reasoning
- Memory for context
- Tools for external capabilities
- Basic planning and execution
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Simulated LLM interface (replace with actual LLM in production)
class LLMInterface:
    """Simulated LLM interface - replace with OpenAI, Anthropic, etc."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Simulate LLM generation - replace with actual API call"""
        # In production, this would call OpenAI, Anthropic, etc.
        print(f"🧠 LLM ({self.model_name}): Processing prompt...")
        
        # Simulate different responses based on prompt content
        if "plan" in prompt.lower():
            return """I'll break this down into steps:
1. Analyze the request
2. Gather necessary information
3. Execute the action
4. Verify the result"""
        elif "search" in prompt.lower():
            return "I need to search for information about this topic."
        elif "calculate" in prompt.lower():
            return "Let me perform the calculation: Result = 42"
        else:
            return "I understand the request and will help you with that."

@dataclass
class AgentMemory:
    """Agent memory system for maintaining context"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 10
    
    def add_interaction(self, role: str, content: str, metadata: Dict = None):
        """Add interaction to short-term memory"""
        interaction = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.short_term.append(interaction)
        
        # Maintain memory limit
        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term summary
            oldest = self.short_term.pop(0)
            self._summarize_to_long_term(oldest)
    
    def _summarize_to_long_term(self, interaction: Dict):
        """Summarize interactions for long-term storage"""
        # Simple summarization - in production, use LLM for this
        summary_key = f"summary_{len(self.long_term)}"
        self.long_term[summary_key] = {
            "timestamp": interaction["timestamp"],
            "summary": f"Interaction about: {interaction['content'][:50]}..."
        }
    
    def get_context(self) -> str:
        """Get formatted context for LLM"""
        context = "Recent interactions:\n"
        for interaction in self.short_term[-5:]:  # Last 5 interactions
            context += f"- {interaction['role']}: {interaction['content']}\n"
        
        if self.long_term:
            context += "\nPrevious context:\n"
            for key, summary in list(self.long_term.items())[-3:]:
                context += f"- {summary['summary']}\n"
        
        return context

class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class SearchTool(Tool):
    """Tool for searching information"""
    
    def name(self) -> str:
        return "search"
    
    def description(self) -> str:
        return "Search for information on the internet or in databases"
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Simulate search functionality"""
        print(f"🔍 Searching for: {query}")
        
        # Simulate search results
        results = [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
            {"title": f"Result 3 for {query}", "url": "https://example.com/3"}
        ]
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "count": len(results)
        }

class CalculatorTool(Tool):
    """Tool for mathematical calculations"""
    
    def name(self) -> str:
        return "calculator"
    
    def description(self) -> str:
        return "Perform mathematical calculations"
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Execute mathematical expression safely"""
        print(f"🧮 Calculating: {expression}")
        
        try:
            # Safe evaluation of mathematical expressions
            # In production, use a proper math parser
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {
                    "success": True,
                    "result": result,
                    "expression": expression
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid characters in expression",
                    "expression": expression
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }

class SimpleAgent:
    """
    Simple Agent implementing the core pattern:
    Perceive → Reason → Act → Adapt
    """
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
        self.llm = LLMInterface()
        self.memory = AgentMemory()
        self.tools = {
            "search": SearchTool(),
            "calculator": CalculatorTool()
        }
        self.conversation_history = []
    
    def perceive(self, user_input: str) -> Dict[str, Any]:
        """Perceive and process user input"""
        print(f"👂 {self.name}: Perceiving input...")
        
        # Add to memory
        self.memory.add_interaction("user", user_input)
        
        # Analyze input
        perception = {
            "input": user_input,
            "timestamp": time.time(),
            "context": self.memory.get_context(),
            "available_tools": list(self.tools.keys())
        }
        
        return perception
    
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about the input and plan actions"""
        print(f"🤔 {self.name}: Reasoning about the request...")
        
        # Create reasoning prompt
        prompt = f"""
        Context: {perception['context']}
        
        User Request: {perception['input']}
        
        Available Tools: {', '.join(perception['available_tools'])}
        
        Please analyze this request and determine:
        1. What the user wants to accomplish
        2. What tools (if any) are needed
        3. What steps should be taken
        
        Respond with a plan.
        """
        
        # Get LLM reasoning
        reasoning_result = self.llm.generate(prompt)
        
        # Determine if tools are needed
        needs_search = "search" in perception['input'].lower() or "find" in perception['input'].lower()
        needs_calculation = any(op in perception['input'] for op in ['+', '-', '*', '/', 'calculate'])
        
        plan = {
            "reasoning": reasoning_result,
            "needs_tools": needs_search or needs_calculation,
            "tools_to_use": [],
            "steps": []
        }
        
        if needs_search:
            plan["tools_to_use"].append("search")
            plan["steps"].append({"action": "search", "query": perception['input']})
        
        if needs_calculation:
            plan["tools_to_use"].append("calculator")
            # Extract mathematical expression (simplified)
            import re
            math_pattern = r'[\d+\-*/().\s]+'
            matches = re.findall(math_pattern, perception['input'])
            if matches:
                plan["steps"].append({"action": "calculator", "expression": matches[0].strip()})
        
        if not plan["steps"]:
            plan["steps"].append({"action": "respond", "message": "I can help you with that."})
        
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned actions"""
        print(f"🎯 {self.name}: Executing plan...")
        
        results = []
        
        for step in plan["steps"]:
            action = step["action"]
            
            if action in self.tools:
                # Use tool
                tool = self.tools[action]
                if action == "search":
                    result = tool.execute(query=step["query"])
                elif action == "calculator":
                    result = tool.execute(expression=step["expression"])
                else:
                    result = tool.execute(**step)
                
                results.append({
                    "action": action,
                    "result": result,
                    "success": result.get("success", True)
                })
            
            elif action == "respond":
                # Direct response
                results.append({
                    "action": "respond",
                    "result": {"message": step["message"]},
                    "success": True
                })
        
        return {
            "plan": plan,
            "results": results,
            "success": all(r["success"] for r in results)
        }
    
    def adapt(self, action_results: Dict[str, Any]) -> str:
        """Adapt based on results and generate final response"""
        print(f"🔄 {self.name}: Adapting and generating response...")
        
        # Create response prompt
        context = self.memory.get_context()
        results_summary = []
        
        for result in action_results["results"]:
            if result["action"] == "search":
                search_data = result["result"]
                if search_data.get("success"):
                    results_summary.append(f"Search found {search_data['count']} results for '{search_data['query']}'")
                else:
                    results_summary.append("Search failed")
            
            elif result["action"] == "calculator":
                calc_data = result["result"]
                if calc_data.get("success"):
                    results_summary.append(f"Calculation result: {calc_data['expression']} = {calc_data['result']}")
                else:
                    results_summary.append(f"Calculation failed: {calc_data.get('error', 'Unknown error')}")
            
            elif result["action"] == "respond":
                results_summary.append(result["result"]["message"])
        
        # Generate final response
        response_prompt = f"""
        Context: {context}
        
        Actions taken: {'; '.join(results_summary)}
        
        Generate a helpful response to the user based on the actions taken and their results.
        """
        
        final_response = self.llm.generate(response_prompt)
        
        # Add to memory
        self.memory.add_interaction("agent", final_response, {
            "actions_taken": [r["action"] for r in action_results["results"]],
            "success": action_results["success"]
        })
        
        return final_response
    
    def process_request(self, user_input: str) -> str:
        """Main processing loop: Perceive → Reason → Act → Adapt"""
        
        print(f"\n🤖 {self.name}: Processing request...")
        print(f"User: {user_input}")
        print("-" * 50)
        
        try:
            # 1. Perceive
            perception = self.perceive(user_input)
            
            # 2. Reason
            plan = self.reason(perception)
            
            # 3. Act
            action_results = self.act(plan)
            
            # 4. Adapt
            response = self.adapt(action_results)
            
            print("-" * 50)
            print(f"Agent: {response}")
            
            return response
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}"
            self.memory.add_interaction("agent", error_response, {"error": True})
            return error_response

def demo_simple_agent():
    """Demonstrate the simple agent capabilities"""
    
    print("🚀 Simple Agent Demo")
    print("=" * 60)
    
    agent = SimpleAgent("DemoAgent")
    
    # Test cases
    test_requests = [
        "What is 15 + 27 * 3?",
        "Search for information about machine learning",
        "Hello, how are you?",
        "Can you help me find articles about Python programming?",
        "Calculate the area of a circle with radius 5 (use 3.14159 for pi)"
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📝 Test {i}/{len(test_requests)}")
        response = agent.process_request(request)
        
        if i < len(test_requests):
            print("\n" + "="*60)
    
    print(f"\n✅ Demo completed! Agent processed {len(test_requests)} requests.")
    print(f"Memory contains {len(agent.memory.short_term)} short-term and {len(agent.memory.long_term)} long-term entries.")

if __name__ == "__main__":
    demo_simple_agent()
