"""
Simple agent that can remember stuff and use tools.
Started this as a weekend project to understand how agents work. And it is still WIP.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# might switch to pydantic later if this gets more complex

class LLMInterface:
    """Mock LLM - will hook up to OpenAI later"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Fake responses for now - saves on API costs during dev"""
        print(f"{self.model_name}: thinking...")
        
        # basic pattern matching until I get the real API working
        if "plan" in prompt.lower():
            return "I'll break this down:\n1. Figure out what you want\n2. Get the info I need\n3. Do the thing\n4. Check if it worked"
        elif "search" in prompt.lower():
            return "I should search for that."
        elif "calculate" in prompt.lower():
            return "Let me do the math... Result = 42"
        else:
            return "Got it, I can help with that."

@dataclass
class AgentMemory:
    """Simple memory system - keeps recent stuff, summarizes old stuff"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 10
    
    def add_interaction(self, role: str, content: str, metadata: Dict = None):
        """Add new interaction to memory"""
        interaction = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.short_term.append(interaction)
        
        # Keep memory from getting too big
        if len(self.short_term) > self.max_short_term:
            oldest = self.short_term.pop(0)
            self._move_to_long_term(oldest)
    
    def _move_to_long_term(self, interaction: Dict):
        """Move old stuff to long term storage"""
        # should probably use the LLM to summarize this better
        summary_key = f"summary_{len(self.long_term)}"
        self.long_term[summary_key] = {
            "timestamp": interaction["timestamp"],
            "summary": f"Talked about: {interaction['content'][:50]}..."
        }
    
    def get_context(self) -> str:
        """Get context string for the LLM"""
        context = "Recent stuff:\n"
        for interaction in self.short_term[-5:]:
            context += f"- {interaction['role']}: {interaction['content']}\n"
        
        if self.long_term:
            context += "\nOlder stuff:\n"
            for key, summary in list(self.long_term.items())[-3:]:
                context += f"- {summary['summary']}\n"
        
        return context

class Tool(ABC):
    """Base class for tools"""
    
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
    """Fake search tool for demo"""
    
    def name(self) -> str:
        return "search"
    
    def description(self) -> str:
        return "Search for stuff"
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Pretend to search"""
        print(f"Searching for: {query}")
        
        # Just make up some results
        results = [
            {"title": f"Some result about {query}", "url": "https://example.com/1"},
            {"title": f"Another {query} result", "url": "https://example.com/2"},
            {"title": f"More {query} stuff", "url": "https://example.com/3"}
        ]
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "count": len(results)
        }

class CalculatorTool(Tool):
    """Basic calculator - yeah I know eval() is sketchy"""
    
    def name(self) -> str:
        return "calculator"
    
    def description(self) -> str:
        return "Do math"
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Calculate stuff - need to replace eval() eventually"""
        print(f"Calculating: {expression}")
        
        try:
            # I know eval() is bad but it's just a prototype
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)  # will use a proper parser later
                return {
                    "success": True,
                    "result": result,
                    "expression": expression
                }
            else:
                return {
                    "success": False,
                    "error": "That looks suspicious",
                    "expression": expression
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Math failed: {str(e)}",
                "expression": expression
            }

class SimpleAgent:
    """Basic agent: perceive, think, act, learn"""
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
        self.llm = LLMInterface()
        self.memory = AgentMemory()
        self.tools = {
            "search": SearchTool(),
            "calculator": CalculatorTool()
        }
    
    def perceive(self, user_input: str) -> Dict[str, Any]:
        """Take in user input and figure out what's going on"""
        print(f"{self.name}: got it...")
        
        self.memory.add_interaction("user", user_input)
        
        return {
            "input": user_input,
            "timestamp": time.time(),
            "context": self.memory.get_context(),
            "available_tools": list(self.tools.keys())
        }
    
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Figure out what to do"""
        print(f"{self.name}: thinking...")
        
        # Ask the LLM what to do
        prompt = f"""
        Context: {perception['context']}
        Request: {perception['input']}
        Tools: {', '.join(perception['available_tools'])}
        
        What should I do?
        """
        
        reasoning_result = self.llm.generate(prompt)
        
        # hacky pattern matching to figure out what tools to use
        user_input = perception['input'].lower()
        needs_search = "search" in user_input or "find" in user_input
        needs_calc = any(op in user_input for op in ['+', '-', '*', '/', 'calculate'])
        
        plan = {
            "reasoning": reasoning_result,
            "steps": []
        }
        
        if needs_search:
            plan["steps"].append({"action": "search", "query": perception['input']})
        
        if needs_calc:
            # try to extract math expression - this is super hacky
            import re
            math_bits = re.findall(r'[\d+\-*/().\s]+', perception['input'])
            if math_bits:
                plan["steps"].append({"action": "calculator", "expression": math_bits[0].strip()})
        
        if not plan["steps"]:
            plan["steps"].append({"action": "respond", "message": "Sure, I can help."})
        
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Do the things"""
        print(f"{self.name}: doing stuff...")
        
        results = []
        
        for step in plan["steps"]:
            action = step["action"]
            
            if action in self.tools:
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
        """Look at what happened and respond"""
        print(f"{self.name}: wrapping up...")
        
        # Summarize what we did
        results_summary = []
        for result in action_results["results"]:
            if result["action"] == "search":
                search_data = result["result"]
                if search_data.get("success"):
                    results_summary.append(f"Found {search_data['count']} results for '{search_data['query']}'")
                else:
                    results_summary.append("Search didn't work")
            
            elif result["action"] == "calculator":
                calc_data = result["result"]
                if calc_data.get("success"):
                    results_summary.append(f"{calc_data['expression']} = {calc_data['result']}")
                else:
                    results_summary.append(f"Math failed: {calc_data.get('error', 'dunno why')}")
            
            elif result["action"] == "respond":
                results_summary.append(result["result"]["message"])
        
        # Get LLM to make a nice response
        response_prompt = f"""
        Context: {self.memory.get_context()}
        What I did: {'; '.join(results_summary)}
        
        Give a helpful response to the user.
        """
        
        final_response = self.llm.generate(response_prompt)
        
        # Remember this interaction
        self.memory.add_interaction("agent", final_response, {
            "actions_taken": [r["action"] for r in action_results["results"]],
            "success": action_results["success"]
        })
        
        return final_response
    
    def process_request(self, user_input: str) -> str:
        """Main loop: perceive → think → act → respond"""
        
        print(f"\n{self.name}: New request")
        print(f"User: {user_input}")
        print("-" * 40)
        
        try:
            perception = self.perceive(user_input)
            plan = self.reason(perception)
            action_results = self.act(plan)
            response = self.adapt(action_results)
            
            print("-" * 40)
            print(f"Agent: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Something broke: {str(e)}"
            self.memory.add_interaction("agent", error_msg, {"error": True})
            return error_msg

def demo_simple_agent():
    """Try out the agent"""
    
    print("Simple Agent Demo")
    print("=" * 50)
    
    agent = SimpleAgent("TestBot")
    
    # Some test cases
    tests = [
        "What is 15 + 27 * 3?",
        "Search for machine learning stuff",
        "Hello there",
        "Find me Python articles",
        "Calculate 3.14159 * 5 * 5"  # circle area
    ]
    
    for i, request in enumerate(tests, 1):
        print(f"\nTest {i}/{len(tests)}")
        agent.process_request(request)
        
        if i < len(tests):
            print("\n" + "="*50)
    
    print(f"\nDone! Processed {len(tests)} requests.")
    print(f"Memory: {len(agent.memory.short_term)} recent, {len(agent.memory.long_term)} old")

if __name__ == "__main__":
    demo_simple_agent()

