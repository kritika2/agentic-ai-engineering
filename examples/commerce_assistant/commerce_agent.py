"""
Commerce Assistant Agent with Temporal Workflows
Implements the commerce assistant example from the article:
- Browses and compares products
- Negotiates deals
- Handles complex multi-step purchasing workflows
- Maintains state across long-running processes
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import Temporal (with fallback for demo)
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    # Mock decorators
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
class Product:
    """Product information"""
    id: str
    name: str
    price: float
    rating: float
    vendor: str
    features: List[str]
    availability: str = "in_stock"

@dataclass
class CommerceRequest:
    """User's commerce request"""
    user_id: str
    request: str
    budget: Optional[float] = None
    preferences: Dict[str, Any] = None
    urgency: str = "normal"  # low, normal, high

@dataclass
class Deal:
    """Negotiated deal information"""
    product: Product
    original_price: float
    negotiated_price: float
    discount_percent: float
    terms: Dict[str, Any]
    expires_at: datetime

# Commerce Activities
@activity.defn
async def browse_products(search_query: str, budget: Optional[float] = None) -> List[Product]:
    """Browse and search for products"""
    print(f"🛒 Browsing products for: {search_query}")
    
    # Simulate product search
    await asyncio.sleep(2)
    
    # Mock product database
    all_products = [
        Product("laptop_1", "MacBook Pro 16\"", 2499.0, 4.8, "Apple", ["M2 Pro", "16GB RAM", "512GB SSD"]),
        Product("laptop_2", "Dell XPS 15", 1899.0, 4.6, "Dell", ["Intel i7", "16GB RAM", "1TB SSD"]),
        Product("laptop_3", "ThinkPad X1 Carbon", 1699.0, 4.7, "Lenovo", ["Intel i7", "16GB RAM", "512GB SSD"]),
        Product("phone_1", "iPhone 15 Pro", 999.0, 4.9, "Apple", ["A17 Pro", "128GB", "Pro Camera"]),
        Product("phone_2", "Samsung Galaxy S24", 899.0, 4.7, "Samsung", ["Snapdragon 8", "256GB", "AI Camera"]),
        Product("headphones_1", "AirPods Pro", 249.0, 4.8, "Apple", ["ANC", "Spatial Audio", "H2 Chip"]),
    ]
    
    # Filter products based on search query
    matching_products = []
    search_lower = search_query.lower()
    
    for product in all_products:
        if (search_lower in product.name.lower() or 
            any(search_lower in feature.lower() for feature in product.features)):
            
            # Apply budget filter if specified
            if budget is None or product.price <= budget:
                matching_products.append(product)
    
    print(f"📦 Found {len(matching_products)} matching products")
    return matching_products[:5]  # Return top 5

@activity.defn
async def compare_products(products: List[Product], preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Compare products and rank them"""
    print(f"⚖️  Comparing {len(products)} products")
    
    await asyncio.sleep(1.5)
    
    if not products:
        return {"rankings": [], "recommendation": None}
    
    # Simple scoring algorithm
    scored_products = []
    
    for product in products:
        score = 0
        
        # Base score from rating
        score += product.rating * 20
        
        # Price preference (lower is better, but not too cheap)
        price_score = max(0, 100 - (product.price / 50))
        score += price_score * 0.3
        
        # Brand preference
        preferred_brands = preferences.get("brands", [])
        if product.vendor in preferred_brands:
            score += 15
        
        # Feature matching
        required_features = preferences.get("features", [])
        feature_matches = sum(1 for req in required_features 
                            if any(req.lower() in feature.lower() for feature in product.features))
        score += feature_matches * 10
        
        scored_products.append({
            "product": product,
            "score": score,
            "price_score": price_score,
            "feature_matches": feature_matches
        })
    
    # Sort by score
    scored_products.sort(key=lambda x: x["score"], reverse=True)
    
    recommendation = scored_products[0] if scored_products else None
    
    return {
        "rankings": scored_products,
        "recommendation": recommendation,
        "comparison_criteria": ["rating", "price", "brand_preference", "feature_matching"]
    }

@activity.defn
async def negotiate_deal(product: Product, user_budget: Optional[float] = None) -> Deal:
    """Attempt to negotiate a better deal"""
    print(f"💰 Negotiating deal for: {product.name}")
    
    await asyncio.sleep(3)  # Negotiation takes time
    
    original_price = product.price
    
    # Negotiation logic
    if user_budget and user_budget < original_price:
        # Try to meet budget
        target_price = user_budget
        discount_percent = ((original_price - target_price) / original_price) * 100
        
        # Limit discount to realistic range
        if discount_percent > 20:
            discount_percent = 20
            target_price = original_price * 0.8
    else:
        # Standard negotiation - try for 5-15% discount
        import random
        discount_percent = random.uniform(5, 15)
        target_price = original_price * (1 - discount_percent / 100)
    
    # Simulate negotiation success rate
    import random
    negotiation_success = random.random() > 0.3  # 70% success rate
    
    if negotiation_success:
        final_price = target_price
        terms = {
            "payment_terms": "30 days",
            "warranty": "extended_1_year",
            "free_shipping": True,
            "return_policy": "30_days"
        }
    else:
        # Partial success
        discount_percent = discount_percent * 0.5
        final_price = original_price * (1 - discount_percent / 100)
        terms = {
            "payment_terms": "standard",
            "warranty": "standard",
            "free_shipping": discount_percent > 5,
            "return_policy": "14_days"
        }
    
    deal = Deal(
        product=product,
        original_price=original_price,
        negotiated_price=final_price,
        discount_percent=discount_percent,
        terms=terms,
        expires_at=datetime.now() + timedelta(hours=24)
    )
    
    print(f"🎯 Negotiated {discount_percent:.1f}% discount: ${original_price:.2f} → ${final_price:.2f}")
    
    return deal

@activity.defn
async def check_inventory(product: Product) -> Dict[str, Any]:
    """Check product inventory and availability"""
    print(f"📋 Checking inventory for: {product.name}")
    
    await asyncio.sleep(1)
    
    # Simulate inventory check
    import random
    
    in_stock = random.random() > 0.1  # 90% chance in stock
    quantity = random.randint(1, 50) if in_stock else 0
    
    estimated_delivery = "2-3 days"
    if quantity < 5:
        estimated_delivery = "5-7 days"
    elif not in_stock:
        estimated_delivery = "2-3 weeks"
    
    return {
        "in_stock": in_stock,
        "quantity_available": quantity,
        "estimated_delivery": estimated_delivery,
        "last_updated": time.time()
    }

@activity.defn
async def process_purchase(deal: Deal, user_id: str) -> Dict[str, Any]:
    """Process the final purchase"""
    print(f"💳 Processing purchase for user: {user_id}")
    
    await asyncio.sleep(2)
    
    # Simulate payment processing
    import random
    
    payment_success = random.random() > 0.05  # 95% success rate
    
    if payment_success:
        order_id = f"ORDER_{int(time.time())}"
        return {
            "success": True,
            "order_id": order_id,
            "amount_charged": deal.negotiated_price,
            "confirmation_number": f"CONF_{order_id}",
            "estimated_delivery": "2-3 days"
        }
    else:
        return {
            "success": False,
            "error": "Payment processing failed",
            "retry_suggested": True
        }

# Main Commerce Workflow
@workflow.defn
class CommerceAgentWorkflow:
    """
    Commerce agent workflow that handles the complete purchasing process
    """
    
    def __init__(self):
        self.retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=3,
            backoff_coefficient=2.0
        ) if TEMPORAL_AVAILABLE else None
    
    @workflow.run
    async def run(self, request: CommerceRequest) -> Dict[str, Any]:
        """Main commerce workflow"""
        
        print(f"🛍️  Starting commerce workflow for user: {request.user_id}")
        print(f"📝 Request: {request.request}")
        
        try:
            # Step 1: Browse products
            if TEMPORAL_AVAILABLE:
                products = await workflow.execute_activity(
                    browse_products,
                    request.request,
                    request.budget,
                    start_to_close_timeout=timedelta(seconds=60)
                )
            else:
                products = await browse_products(request.request, request.budget)
            
            if not products:
                return {
                    "success": False,
                    "message": "No products found matching your criteria",
                    "user_id": request.user_id
                }
            
            # Step 2: Compare products
            preferences = request.preferences or {}
            
            if TEMPORAL_AVAILABLE:
                comparison = await workflow.execute_activity(
                    compare_products,
                    products,
                    preferences,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                comparison = await compare_products(products, preferences)
            
            if not comparison["recommendation"]:
                return {
                    "success": False,
                    "message": "Could not find suitable product recommendations",
                    "products_found": len(products)
                }
            
            recommended_product = comparison["recommendation"]["product"]
            
            # Step 3: Check inventory
            if TEMPORAL_AVAILABLE:
                inventory = await workflow.execute_activity(
                    check_inventory,
                    recommended_product,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                inventory = await check_inventory(recommended_product)
            
            if not inventory["in_stock"]:
                return {
                    "success": False,
                    "message": f"Sorry, {recommended_product.name} is currently out of stock",
                    "estimated_restock": inventory["estimated_delivery"],
                    "alternative_products": [p.name for p in products[1:3]]
                }
            
            # Step 4: Negotiate deal
            if TEMPORAL_AVAILABLE:
                deal = await workflow.execute_activity(
                    negotiate_deal,
                    recommended_product,
                    request.budget,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=self.retry_policy
                )
            else:
                deal = await negotiate_deal(recommended_product, request.budget)
            
            # Step 5: Process purchase (if urgent or auto-approved)
            if request.urgency == "high" or deal.discount_percent > 10:
                if TEMPORAL_AVAILABLE:
                    purchase_result = await workflow.execute_activity(
                        process_purchase,
                        deal,
                        request.user_id,
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=self.retry_policy
                    )
                else:
                    purchase_result = await process_purchase(deal, request.user_id)
                
                return {
                    "success": True,
                    "action": "purchase_completed",
                    "deal": {
                        "product_name": deal.product.name,
                        "original_price": deal.original_price,
                        "final_price": deal.negotiated_price,
                        "discount_percent": deal.discount_percent,
                        "savings": deal.original_price - deal.negotiated_price
                    },
                    "purchase": purchase_result,
                    "user_id": request.user_id
                }
            else:
                # Return deal for user approval
                return {
                    "success": True,
                    "action": "deal_ready_for_approval",
                    "deal": {
                        "product_name": deal.product.name,
                        "original_price": deal.original_price,
                        "final_price": deal.negotiated_price,
                        "discount_percent": deal.discount_percent,
                        "savings": deal.original_price - deal.negotiated_price,
                        "terms": deal.terms,
                        "expires_at": deal.expires_at.isoformat()
                    },
                    "inventory": inventory,
                    "user_id": request.user_id,
                    "message": "Great deal found! Please review and approve the purchase."
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": request.user_id,
                "message": "An error occurred while processing your request"
            }

async def demo_commerce_agent():
    """Demonstrate the commerce agent workflow"""
    
    print("🛍️  Commerce Assistant Agent Demo")
    print("=" * 60)
    
    # Sample requests
    requests = [
        CommerceRequest(
            user_id="user_001",
            request="I need a high-performance laptop for software development",
            budget=2000.0,
            preferences={
                "brands": ["Apple", "Dell"],
                "features": ["16GB RAM", "SSD"]
            },
            urgency="normal"
        ),
        CommerceRequest(
            user_id="user_002", 
            request="Looking for wireless headphones with noise cancellation",
            budget=300.0,
            preferences={
                "features": ["ANC", "Bluetooth"]
            },
            urgency="high"
        )
    ]
    
    for i, request in enumerate(requests, 1):
        print(f"\n🔄 Processing Request {i}/{len(requests)}")
        print(f"User: {request.user_id}")
        print(f"Request: {request.request}")
        print(f"Budget: ${request.budget}")
        print("-" * 40)
        
        # Create workflow instance
        workflow_instance = CommerceAgentWorkflow()
        
        # Execute workflow
        result = await workflow_instance.run(request)
        
        # Display results
        print(f"\n📊 Result:")
        if result["success"]:
            if result["action"] == "purchase_completed":
                print(f"✅ Purchase completed!")
                deal = result["deal"]
                print(f"   Product: {deal['product_name']}")
                print(f"   Price: ${deal['original_price']:.2f} → ${deal['final_price']:.2f}")
                print(f"   Savings: ${deal['savings']:.2f} ({deal['discount_percent']:.1f}% off)")
                
                if result["purchase"]["success"]:
                    print(f"   Order ID: {result['purchase']['order_id']}")
                    print(f"   Delivery: {result['purchase']['estimated_delivery']}")
            
            elif result["action"] == "deal_ready_for_approval":
                print(f"💼 Deal ready for approval:")
                deal = result["deal"]
                print(f"   Product: {deal['product_name']}")
                print(f"   Price: ${deal['original_price']:.2f} → ${deal['final_price']:.2f}")
                print(f"   Savings: ${deal['savings']:.2f} ({deal['discount_percent']:.1f}% off)")
                print(f"   Expires: {deal['expires_at']}")
        else:
            print(f"❌ {result['message']}")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        if i < len(requests):
            print("\n" + "=" * 60)
    
    print(f"\n✅ Demo completed! Processed {len(requests)} commerce requests.")

if __name__ == "__main__":
    asyncio.run(demo_commerce_agent())
