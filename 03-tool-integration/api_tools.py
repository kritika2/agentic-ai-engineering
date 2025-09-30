"""
API tools for agents to talk to external services.
Spent way too much time getting the retry logic right.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import aiohttp
import requests
from datetime import datetime, timedelta

@dataclass
class APIConfig:
    """Configuration for API integration"""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

class APITool(ABC):
    """Base class for API tools"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the API call"""
        pass
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retries and error handling"""
        
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {**self.config.headers}
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        for attempt in range(self.config.max_retries):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.request(
                    method, url, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    **kwargs
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return {"success": True, "data": data, "status": response.status}
                    
                    elif response.status in [429, 500, 502, 503, 504]:  # retry these
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                    
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "status": response.status
                    }
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                return {"success": False, "error": "Request timeout"}
            
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                return {"success": False, "error": f"Request failed: {str(e)}"}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class WeatherAPI(APITool):
    """Weather API integration"""
    
    def __init__(self, api_key: str):
        config = APIConfig(
            base_url="https://api.openweathermap.org/data/2.5",
            api_key=api_key,
            timeout=10
        )
        super().__init__(config)
    
    async def execute(self, city: str, units: str = "metric") -> Dict[str, Any]:
        """Get weather for a city"""
        
        params = {
            "q": city,
            "appid": self.config.api_key,
            "units": units
        }
        
        result = await self._make_request("GET", "weather", params=params)
        
        if result["success"]:
            weather_data = result["data"]
            return {
                "success": True,
                "city": city,
                "temperature": weather_data["main"]["temp"],
                "description": weather_data["weather"][0]["description"],
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": weather_data["wind"]["speed"]
            }
        else:
            return result

class NewsAPI(APITool):
    """News API integration"""
    
    def __init__(self, api_key: str):
        config = APIConfig(
            base_url="https://newsapi.org/v2",
            api_key=api_key,
            timeout=15
        )
        super().__init__(config)
    
    async def execute(self, query: str, language: str = "en", page_size: int = 10) -> Dict[str, Any]:
        """Search for news articles"""
        
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "apiKey": self.config.api_key
        }
        
        result = await self._make_request("GET", "everything", params=params)
        
        if result["success"]:
            news_data = result["data"]
            articles = []
            
            for article in news_data.get("articles", []):
                articles.append({
                    "title": article["title"],
                    "description": article["description"],
                    "url": article["url"],
                    "published_at": article["publishedAt"],
                    "source": article["source"]["name"]
                })
            
            return {
                "success": True,
                "query": query,
                "total_results": news_data.get("totalResults", 0),
                "articles": articles
            }
        else:
            return result

class MockCRMAPI(APITool):
    """Fake CRM API since I don't have a real one to test with"""
    
    def __init__(self):
        config = APIConfig(
            base_url="https://mock-crm-api.example.com/v1",
            timeout=5
        )
        super().__init__(config)
        self.fake_customers = {
            "customers": [
                {"id": 1, "name": "John Doe", "email": "john@example.com", "status": "active"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "status": "inactive"},
                {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "status": "active"}
            ]
        }
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute CRM operations"""
        
        # fake implementation since I don't have a real CRM to test with
        await asyncio.sleep(0.5)  # pretend there's network delay
        
        if action == "get_customers":
            status_filter = kwargs.get("status")
            customers = self.fake_customers["customers"]
            
            if status_filter:
                customers = [c for c in customers if c["status"] == status_filter]
            
            return {
                "success": True,
                "action": action,
                "customers": customers,
                "count": len(customers)
            }
        
        elif action == "create_customer":
            new_customer = {
                "id": len(self.fake_customers["customers"]) + 1,
                "name": kwargs.get("name", "Unknown"),
                "email": kwargs.get("email", "unknown@example.com"),
                "status": "active"
            }
            self.fake_customers["customers"].append(new_customer)
            
            return {
                "success": True,
                "action": action,
                "customer": new_customer
            }
        
        elif action == "update_customer":
            customer_id = kwargs.get("id")
            for customer in self.fake_customers["customers"]:
                if customer["id"] == customer_id:
                    customer.update(kwargs)
                    return {
                        "success": True,
                        "action": action,
                        "customer": customer
                    }
            
            return {
                "success": False,
                "error": f"Customer {customer_id} not found"
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }

class DatabaseTool:
    """Database tool - using fake data for now"""
    
    def __init__(self, connection_string: str = "sqlite:///:memory:"):
        self.connection_string = connection_string
        self.fake_data = {
            "users": [
                {"id": 1, "username": "alice", "email": "alice@example.com", "created_at": "2024-01-01"},
                {"id": 2, "username": "bob", "email": "bob@example.com", "created_at": "2024-01-02"},
                {"id": 3, "username": "charlie", "email": "charlie@example.com", "created_at": "2024-01-03"}
            ],
            "orders": [
                {"id": 1, "user_id": 1, "amount": 99.99, "status": "completed"},
                {"id": 2, "user_id": 2, "amount": 149.99, "status": "pending"},
                {"id": 3, "user_id": 1, "amount": 79.99, "status": "completed"}
            ]
        }
    
    async def execute(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """Execute database operations"""
        
        await asyncio.sleep(0.2)  # pretend the database is slow
        
        if query_type == "select_users":
            users = self.fake_data["users"]
            limit = kwargs.get("limit", len(users))
            
            return {
                "success": True,
                "query_type": query_type,
                "results": users[:limit],
                "count": len(users[:limit])
            }
        
        elif query_type == "select_orders":
            orders = self.fake_data["orders"]
            user_id = kwargs.get("user_id")
            
            if user_id:
                orders = [o for o in orders if o["user_id"] == user_id]
            
            return {
                "success": True,
                "query_type": query_type,
                "results": orders,
                "count": len(orders)
            }
        
        elif query_type == "aggregate_sales":
            orders = self.fake_data["orders"]
            completed_orders = [o for o in orders if o["status"] == "completed"]
            total_sales = sum(o["amount"] for o in completed_orders)
            
            return {
                "success": True,
                "query_type": query_type,
                "total_sales": total_sales,
                "completed_orders": len(completed_orders),
                "total_orders": len(orders)
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown query type: {query_type}"
            }

class ToolIntegrationAgent:
    """Agent that uses multiple tools"""
    
    def __init__(self, name: str = "ToolAgent"):
        self.name = name
        self.tools = {}
        self.execution_history = []
    
    def add_tool(self, name: str, tool):
        """Add a tool to the agent"""
        self.tools[name] = tool
        print(f"Added tool: {name}")
    
    async def execute_task(self, task_description: str, tool_preferences: List[str] = None) -> Dict[str, Any]:
        """Execute a task using available tools"""
        
        print(f"\n{self.name}: Executing task")
        print(f"Task: {task_description}")
        print(f"Available tools: {list(self.tools.keys())}")
        print("-" * 50)
        
        start_time = time.time()
        results = {}
        
        try:
            # basic keyword matching to figure out what tools to use
            task_lower = task_description.lower()
            
            if "weather" in task_lower and "weather" in self.tools:
                # Extract city from task description
                words = task_description.split()
                city = "London"  # Default
                for i, word in enumerate(words):
                    if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                        city = words[i + 1]
                        break
                
                weather_result = await self.tools["weather"].execute(city=city)
                results["weather"] = weather_result
            
            if "news" in task_lower and "news" in self.tools:
                # Extract search query
                query = task_description
                if "about" in task_lower:
                    query = task_description.split("about", 1)[1].strip()
                
                news_result = await self.tools["news"].execute(query=query)
                results["news"] = news_result
            
            if "customer" in task_lower and "crm" in self.tools:
                if "get" in task_lower or "list" in task_lower:
                    crm_result = await self.tools["crm"].execute(action="get_customers")
                elif "create" in task_lower:
                    crm_result = await self.tools["crm"].execute(
                        action="create_customer",
                        name="New Customer",
                        email="new@example.com"
                    )
                else:
                    crm_result = await self.tools["crm"].execute(action="get_customers")
                
                results["crm"] = crm_result
            
            if "database" in task_lower or "users" in task_lower or "orders" in task_lower and "db" in self.tools:
                if "users" in task_lower:
                    db_result = await self.tools["db"].execute(query_type="select_users", limit=5)
                elif "orders" in task_lower:
                    db_result = await self.tools["db"].execute(query_type="select_orders")
                elif "sales" in task_lower:
                    db_result = await self.tools["db"].execute(query_type="aggregate_sales")
                else:
                    db_result = await self.tools["db"].execute(query_type="select_users")
                
                results["database"] = db_result
            
            # if nothing matched, just try whatever tools we have
            if not results and self.tools:
                print("No specific tool match, trying available tools...")
                for tool_name, tool in list(self.tools.items())[:2]:  # only try first 2
                    try:
                        if hasattr(tool, 'execute'):
                            if tool_name == "weather":
                                result = await tool.execute(city="London")
                            elif tool_name == "news":
                                result = await tool.execute(query="technology")
                            elif tool_name == "crm":
                                result = await tool.execute(action="get_customers")
                            elif tool_name == "db":
                                result = await tool.execute(query_type="select_users")
                            else:
                                continue
                            
                            results[tool_name] = result
                    except Exception as e:
                        print(f"Tool {tool_name} failed: {str(e)}")
            
            execution_time = time.time() - start_time
            
            # Summarize results
            summary = {
                "task": task_description,
                "tools_used": list(results.keys()),
                "success": any(r.get("success", False) for r in results.values()),
                "results": results,
                "execution_time": execution_time
            }
            
            self.execution_history.append(summary)
            
            print(f"\nTask completed in {execution_time:.2f}s")
            print(f"Tools used: {', '.join(results.keys())}")
            
            for tool_name, result in results.items():
                if result.get("success"):
                    print(f"{tool_name}: Success")
                else:
                    print(f"{tool_name}: {result.get('error', 'Failed')}")
            
            return summary
            
        except Exception as e:
            error_summary = {
                "task": task_description,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            self.execution_history.append(error_summary)
            print(f"Task failed: {str(e)}")
            return error_summary
    
    async def cleanup(self):
        """Clean up tool resources"""
        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                await tool.close()

async def demo_tool_integration():
    """Demo the tool integration system"""
    
    print("Tool Integration Demo")
    print("=" * 60)
    
    # Create agent
    agent = ToolIntegrationAgent("ToolBot")
    
    # Add tools (using mock APIs for demo)
    agent.add_tool("weather", WeatherAPI("demo_key"))  # Would need real API key
    agent.add_tool("news", NewsAPI("demo_key"))        # Would need real API key
    agent.add_tool("crm", MockCRMAPI())
    agent.add_tool("db", DatabaseTool())
    
    # Test tasks
    test_tasks = [
        "Get the weather in New York",
        "Find news about artificial intelligence",
        "List all customers from CRM",
        "Get user data from database",
        "Show me sales statistics",
        "Create a new customer in the system"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"TASK {i}/{len(test_tasks)}")
        
        result = await agent.execute_task(task)
        
        if i < len(test_tasks):
            print(f"\nContinuing to next task...")
            time.sleep(1)
    
    # Cleanup
    await agent.cleanup()
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"Tasks executed: {len(agent.execution_history)}")
    print(f"Successful: {sum(1 for h in agent.execution_history if h['success'])}")

if __name__ == "__main__":
    asyncio.run(demo_tool_integration())
