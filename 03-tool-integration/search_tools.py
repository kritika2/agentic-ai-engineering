"""
Search and retrieval tools for agents.
Web search, document search, vector search, and knowledge retrieval.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import re

@dataclass
class SearchResult:
    """A search result"""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Document:
    """A document for search/retrieval"""
    id: str
    title: str
    content: str
    url: Optional[str] = None
    tags: List[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class WebSearchTool:
    """Web search tool (mock implementation)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.search_history = []
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search the web"""
        
        print(f"Searching web for: {query}")
        await asyncio.sleep(1)  # Simulate search time
        
        # Mock search results - in real implementation would use Google/Bing API
        mock_results = [
            SearchResult(
                title=f"Understanding {query} - Complete Guide",
                url=f"https://example.com/guide-{query.replace(' ', '-')}",
                snippet=f"A comprehensive guide to {query} covering all the basics and advanced concepts.",
                score=0.95,
                source="example.com"
            ),
            SearchResult(
                title=f"{query} Best Practices",
                url=f"https://bestpractices.com/{query.replace(' ', '-')}",
                snippet=f"Learn the best practices for {query} from industry experts.",
                score=0.88,
                source="bestpractices.com"
            ),
            SearchResult(
                title=f"Latest News on {query}",
                url=f"https://news.com/latest-{query.replace(' ', '-')}",
                snippet=f"Recent developments and news about {query}.",
                score=0.82,
                source="news.com"
            ),
            SearchResult(
                title=f"{query} Tutorial for Beginners",
                url=f"https://tutorials.com/{query.replace(' ', '-')}-tutorial",
                snippet=f"Step-by-step tutorial for beginners learning about {query}.",
                score=0.79,
                source="tutorials.com"
            ),
            SearchResult(
                title=f"Advanced {query} Techniques",
                url=f"https://advanced.com/{query.replace(' ', '-')}-advanced",
                snippet=f"Advanced techniques and strategies for {query}.",
                score=0.75,
                source="advanced.com"
            )
        ]
        
        # Filter to requested number of results
        results = mock_results[:num_results]
        
        # Store search history
        self.search_history.append({
            "query": query,
            "num_results": len(results),
            "timestamp": time.time()
        })
        
        return results
    
    async def scrape_content(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL"""
        
        print(f"Scraping content from: {url}")
        await asyncio.sleep(0.5)
        
        # Mock scraping - in real implementation would use requests + BeautifulSoup
        mock_content = f"""
        This is the scraped content from {url}.
        
        It contains detailed information about the topic, including:
        - Key concepts and definitions
        - Practical examples and use cases
        - Best practices and recommendations
        - Common pitfalls to avoid
        
        The content is well-structured and provides valuable insights
        for anyone looking to understand this topic better.
        """
        
        return {
            "success": True,
            "url": url,
            "title": f"Content from {url}",
            "content": mock_content.strip(),
            "word_count": len(mock_content.split()),
            "scraped_at": time.time()
        }

class DocumentSearchTool:
    """Search through a collection of documents"""
    
    def __init__(self):
        self.documents = self._create_mock_documents()
        self.search_index = self._build_search_index()
    
    def _create_mock_documents(self) -> List[Document]:
        """Create mock documents for demo"""
        
        return [
            Document(
                id="doc_1",
                title="Introduction to Machine Learning",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
                tags=["AI", "ML", "algorithms", "data science"],
                created_at="2024-01-01"
            ),
            Document(
                id="doc_2", 
                title="Python Programming Best Practices",
                content="Python is a versatile programming language. Best practices include writing clean code, using virtual environments, following PEP 8 style guidelines, and writing comprehensive tests.",
                tags=["Python", "programming", "best practices", "coding"],
                created_at="2024-01-02"
            ),
            Document(
                id="doc_3",
                title="Database Design Principles",
                content="Good database design involves normalization, proper indexing, and understanding relationships between entities. Consider performance, scalability, and data integrity.",
                tags=["database", "design", "SQL", "normalization"],
                created_at="2024-01-03"
            ),
            Document(
                id="doc_4",
                title="Web Development with APIs",
                content="Modern web development relies heavily on APIs. RESTful APIs provide a standard way to communicate between services. Consider authentication, rate limiting, and error handling.",
                tags=["web development", "APIs", "REST", "backend"],
                created_at="2024-01-04"
            ),
            Document(
                id="doc_5",
                title="Cloud Computing Fundamentals",
                content="Cloud computing provides on-demand access to computing resources. Key concepts include Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).",
                tags=["cloud", "AWS", "infrastructure", "scalability"],
                created_at="2024-01-05"
            )
        ]
    
    def _build_search_index(self) -> Dict[str, List[str]]:
        """Build a simple search index"""
        
        index = {}
        
        for doc in self.documents:
            # Index title and content words
            words = (doc.title + " " + doc.content).lower().split()
            words = [re.sub(r'[^\w]', '', word) for word in words if len(word) > 2]
            
            for word in words:
                if word not in index:
                    index[word] = []
                if doc.id not in index[word]:
                    index[word].append(doc.id)
        
        return index
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search documents"""
        
        print(f"Searching documents for: {query}")
        await asyncio.sleep(0.3)
        
        query_words = [re.sub(r'[^\w]', '', word.lower()) for word in query.split() if len(word) > 2]
        
        # Score documents based on word matches
        doc_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for doc_id in self.search_index[word]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1
        
        # Sort by score and convert to SearchResult objects
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:max_results]:
            doc = next(d for d in self.documents if d.id == doc_id)
            
            # Create snippet from content
            snippet = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
            
            results.append(SearchResult(
                title=doc.title,
                url=f"/documents/{doc.id}",
                snippet=snippet,
                score=score / len(query_words),  # Normalize score
                source="document_collection",
                metadata={"doc_id": doc.id, "tags": doc.tags}
            ))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a specific document"""
        return next((d for d in self.documents if d.id == doc_id), None)

class VectorSearchTool:
    """Mock vector search tool (would use embeddings in real implementation)"""
    
    def __init__(self):
        self.vectors = self._create_mock_vectors()
    
    def _create_mock_vectors(self) -> Dict[str, Dict[str, Any]]:
        """Create mock vector data"""
        
        return {
            "ai_concepts": {
                "content": "Artificial intelligence, machine learning, deep learning, neural networks",
                "vector": [0.8, 0.6, 0.9, 0.7, 0.5],  # Mock embedding
                "metadata": {"category": "AI", "difficulty": "intermediate"}
            },
            "programming": {
                "content": "Python, JavaScript, coding, software development, algorithms",
                "vector": [0.7, 0.8, 0.6, 0.9, 0.4],
                "metadata": {"category": "Programming", "difficulty": "beginner"}
            },
            "databases": {
                "content": "SQL, NoSQL, database design, data modeling, indexing",
                "vector": [0.6, 0.7, 0.8, 0.5, 0.9],
                "metadata": {"category": "Database", "difficulty": "intermediate"}
            },
            "web_dev": {
                "content": "HTML, CSS, JavaScript, React, APIs, web development",
                "vector": [0.5, 0.9, 0.7, 0.8, 0.6],
                "metadata": {"category": "Web Development", "difficulty": "beginner"}
            },
            "cloud": {
                "content": "AWS, Azure, cloud computing, serverless, containers",
                "vector": [0.9, 0.5, 0.6, 0.7, 0.8],
                "metadata": {"category": "Cloud", "difficulty": "advanced"}
            }
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _query_to_vector(self, query: str) -> List[float]:
        """Convert query to mock vector"""
        
        # Simple mock: hash query to generate consistent vector
        import hashlib
        hash_obj = hashlib.md5(query.encode())
        hash_bytes = hash_obj.digest()[:5]  # Take first 5 bytes
        
        # Convert to normalized vector
        vector = [b / 255.0 for b in hash_bytes]
        return vector
    
    async def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Vector similarity search"""
        
        print(f"Vector searching for: {query}")
        await asyncio.sleep(0.5)
        
        query_vector = self._query_to_vector(query)
        
        # Calculate similarities
        similarities = []
        for key, data in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, data["vector"])
            similarities.append((key, similarity, data))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to SearchResult objects
        results = []
        for key, similarity, data in similarities[:max_results]:
            results.append(SearchResult(
                title=f"Vector Match: {data['metadata']['category']}",
                url=f"/vectors/{key}",
                snippet=data["content"],
                score=similarity,
                source="vector_search",
                metadata=data["metadata"]
            ))
        
        return results

class KnowledgeRetrievalAgent:
    """Agent that combines multiple search tools"""
    
    def __init__(self, name: str = "SearchAgent"):
        self.name = name
        self.web_search = WebSearchTool()
        self.doc_search = DocumentSearchTool()
        self.vector_search = VectorSearchTool()
        self.search_history = []
    
    async def comprehensive_search(self, query: str, search_types: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive search across multiple sources"""
        
        if search_types is None:
            search_types = ["web", "documents", "vectors"]
        
        print(f"\n{self.name}: Comprehensive search")
        print(f"Query: {query}")
        print(f"Search types: {', '.join(search_types)}")
        print("-" * 50)
        
        start_time = time.time()
        results = {}
        
        try:
            # Web search
            if "web" in search_types:
                web_results = await self.web_search.search(query, num_results=5)
                results["web"] = {
                    "results": web_results,
                    "count": len(web_results),
                    "source": "web_search"
                }
            
            # Document search
            if "documents" in search_types:
                doc_results = await self.doc_search.search(query, max_results=3)
                results["documents"] = {
                    "results": doc_results,
                    "count": len(doc_results),
                    "source": "document_collection"
                }
            
            # Vector search
            if "vectors" in search_types:
                vector_results = await self.vector_search.search(query, max_results=3)
                results["vectors"] = {
                    "results": vector_results,
                    "count": len(vector_results),
                    "source": "vector_database"
                }
            
            # Combine and rank all results
            all_results = []
            for search_type, data in results.items():
                for result in data["results"]:
                    result.metadata["search_type"] = search_type
                    all_results.append(result)
            
            # Sort by score
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            execution_time = time.time() - start_time
            
            summary = {
                "query": query,
                "search_types": search_types,
                "total_results": len(all_results),
                "results_by_type": results,
                "top_results": all_results[:10],
                "execution_time": execution_time,
                "success": True
            }
            
            self.search_history.append(summary)
            
            print(f"\nSearch completed in {execution_time:.2f}s")
            print(f"Total results: {len(all_results)}")
            
            for search_type, data in results.items():
                print(f"  {search_type}: {data['count']} results")
            
            print(f"\nTop results:")
            for i, result in enumerate(all_results[:5], 1):
                print(f"  {i}. {result.title} (score: {result.score:.2f}, source: {result.metadata.get('search_type', 'unknown')})")
            
            return summary
            
        except Exception as e:
            error_summary = {
                "query": query,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            self.search_history.append(error_summary)
            print(f"Search failed: {str(e)}")
            return error_summary
    
    async def get_detailed_content(self, search_result: SearchResult) -> Dict[str, Any]:
        """Get detailed content for a search result"""
        
        if search_result.metadata.get("search_type") == "web":
            return await self.web_search.scrape_content(search_result.url)
        
        elif search_result.metadata.get("search_type") == "documents":
            doc_id = search_result.metadata.get("doc_id")
            if doc_id:
                doc = self.doc_search.get_document(doc_id)
                if doc:
                    return {
                        "success": True,
                        "title": doc.title,
                        "content": doc.content,
                        "tags": doc.tags,
                        "created_at": doc.created_at
                    }
        
        return {"success": False, "error": "Content not available"}

async def demo_search_tools():
    """Demo the search and retrieval system"""
    
    print("Search and Retrieval Tools Demo")
    print("=" * 60)
    
    # Create agent
    agent = KnowledgeRetrievalAgent("SearchBot")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "Python programming best practices", 
        "database design principles",
        "web development with APIs",
        "cloud computing fundamentals"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"SEARCH {i}/{len(test_queries)}")
        
        result = await agent.comprehensive_search(query)
        
        if result["success"] and result["top_results"]:
            print(f"\nGetting detailed content for top result...")
            top_result = result["top_results"][0]
            detailed_content = await agent.get_detailed_content(top_result)
            
            if detailed_content["success"]:
                print(f"Retrieved detailed content ({detailed_content.get('word_count', 'unknown')} words)")
            else:
                print(f"Could not retrieve detailed content")
        
        if i < len(test_queries):
            print(f"\nContinuing to next search...")
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"Searches performed: {len(agent.search_history)}")
    print(f"Successful: {sum(1 for h in agent.search_history if h['success'])}")

if __name__ == "__main__":
    asyncio.run(demo_search_tools())
