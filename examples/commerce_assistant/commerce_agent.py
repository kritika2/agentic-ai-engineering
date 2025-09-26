"""
Commerce bot that can browse products, compare stuff, and negotiate deals.
Uses Temporal so it doesn't lose track of what it's doing.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# This could probably be refactored into smaller modules

# Temporal imports with fallback
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    # Mock classes for when Temporal isn't available
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
    """A product"""
    id: str
    name: str
    price: float
    rating: float
    vendor: str
    features: List[str]
    availability: str = "in_stock"

@dataclass
class CommerceRequest:
    """What the user wants"""
    user_id: str
    request: str
    budget: Optional[float] = None
    preferences: Dict[str, Any] = None
    urgency: str = "normal"  # low, normal, high

@dataclass
class Deal:
    """A negotiated deal"""
    product: Product
    original_price: float
    negotiated_price: float
    discount_percent: float
    terms: Dict[str, Any]
    expires_at: datetime

# Activities
@activity.defn
async def browse_products(search_query: str, budget: Optional[float] = None) -> List[Product]:
    """Find products"""
    print(f"Looking for: {search_query}")
    
    await asyncio.sleep(2)  # fake search time
    
    # Fake product database
    products = [
        Product("laptop_1", "MacBook Pro 16\"", 2499.0, 4.8, "Apple", ["M2 Pro", "16GB RAM", "512GB SSD"]),
        Product("laptop_2", "Dell XPS 15", 1899.0, 4.6, "Dell", ["Intel i7", "16GB RAM", "1TB SSD"]),
        Product("laptop_3", "ThinkPad X1 Carbon", 1699.0, 4.7, "Lenovo", ["Intel i7", "16GB RAM", "512GB SSD"]),
        Product("phone_1", "iPhone 15 Pro", 999.0, 4.9, "Apple", ["A17 Pro", "128GB", "Pro Camera"]),
        Product("phone_2", "Samsung Galaxy S24", 899.0, 4.7, "Samsung", ["Snapdragon 8", "256GB", "AI Camera"]),
        Product("headphones_1", "AirPods Pro", 249.0, 4.8, "Apple", ["ANC", "Spatial Audio", "H2 Chip"]),
    ]
    
    # Basic search
    matches = []
    search_lower = search_query.lower()
    
    for product in products:
        if (search_lower in product.name.lower() or 
            any(search_lower in feature.lower() for feature in product.features)):
            if budget is None or product.price <= budget:
                matches.append(product)
    
    print(f"Found {len(matches)} products")
    return matches[:5]

@activity.defn
async def compare_products(products: List[Product], preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Compare and rank products"""
    print(f"Comparing {len(products)} products")
    
    await asyncio.sleep(1.5)
    
    if not products:
        return {"rankings": [], "recommendation": None}
    
    # Simple scoring - this could be way more sophisticated
    scored = []
    
    for product in products:
        score = product.rating * 20  # base score
        
        # Cheaper is better (kinda)
        price_score = max(0, 100 - (product.price / 50))
        score += price_score * 0.3
        
        # Brand preferences
        if product.vendor in preferences.get("brands", []):
            score += 15
        
        # Feature matching
        wanted_features = preferences.get("features", [])
        feature_matches = sum(1 for req in wanted_features 
                            if any(req.lower() in feature.lower() for feature in product.features))
        score += feature_matches * 10
        
        scored.append({
            "product": product,
            "score": score,
            "feature_matches": feature_matches
        })
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "rankings": scored,
        "recommendation": scored[0] if scored else None
    }

@activity.defn
async def negotiate_deal(product: Product, user_budget: Optional[float] = None) -> Deal:
    """Try to get a better price"""
    print(f"Negotiating for: {product.name}")
    
    await asyncio.sleep(3)  # negotiations take time
    
    original_price = product.price
    
    # Figure out target price
    if user_budget and user_budget < original_price:
        target_price = user_budget
        discount_percent = ((original_price - target_price) / original_price) * 100
        if discount_percent > 20:  # max 20% discount
            discount_percent = 20
            target_price = original_price * 0.8
    else:
        # Try for 5-15% off
        import random
        discount_percent = random.uniform(5, 15)
        target_price = original_price * (1 - discount_percent / 100)
    
    # Sometimes negotiations fail
    import random
    success = random.random() > 0.3  # 70% success
    
    if success:
        final_price = target_price
        terms = {"payment_terms": "30 days", "warranty": "extended", "free_shipping": True}
    else:
        # Partial success
        discount_percent *= 0.5
        final_price = original_price * (1 - discount_percent / 100)
        terms = {"payment_terms": "standard", "warranty": "standard", "free_shipping": False}
    
    deal = Deal(
        product=product, original_price=original_price, negotiated_price=final_price,
        discount_percent=discount_percent, terms=terms,
        expires_at=datetime.now() + timedelta(hours=24)
    )
    
    print(f"Got {discount_percent:.1f}% off: ${original_price:.2f} → ${final_price:.2f}")
    return deal

@activity.defn
async def check_inventory(product: Product) -> Dict[str, Any]:
    """Check if we have it in stock"""
    print(f"Checking stock: {product.name}")
    
    await asyncio.sleep(1)
    
    import random
    in_stock = random.random() > 0.1  # usually in stock
    quantity = random.randint(1, 50) if in_stock else 0
    
    delivery = "2-3 days"
    if quantity < 5:
        delivery = "5-7 days"  # low stock
    elif not in_stock:
        delivery = "2-3 weeks"  # out of stock
    
    return {
        "in_stock": in_stock,
        "quantity_available": quantity,
        "estimated_delivery": delivery,
        "last_updated": time.time()
    }

@activity.defn
async def process_purchase(deal: Deal, user_id: str) -> Dict[str, Any]:
    """Actually buy the thing"""
    print(f"Processing purchase for: {user_id}")
    
    await asyncio.sleep(2)
    
    import random
    payment_works = random.random() > 0.05  # 95% success
    
    if payment_works:
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
            "error": "Payment failed - try again?",
            "retry_suggested": True
        }

# Main workflow
@workflow.defn
class CommerceAgentWorkflow:
    """Commerce agent that buys stuff"""
    
    def __init__(self):
        self.retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=3,
            backoff_coefficient=2.0
        ) if TEMPORAL_AVAILABLE else None
    
    @workflow.run
    async def run(self, request: CommerceRequest) -> Dict[str, Any]:
        """Do the commerce thing"""
        
        print(f"Starting for user: {request.user_id}")
        print(f"They want: {request.request}")
        
        try:
            # Step 1: Find products
            if TEMPORAL_AVAILABLE:
                products = await workflow.execute_activity(
                    browse_products, request.request, request.budget,
                    start_to_close_timeout=timedelta(seconds=60)
                )
            else:
                products = await browse_products(request.request, request.budget)
            
            if not products:
                return {
                    "success": False,
                    "message": "Couldn't find anything matching that",
                    "user_id": request.user_id
                }
            
            # Step 2: Compare them
            preferences = request.preferences or {}
            
            if TEMPORAL_AVAILABLE:
                comparison = await workflow.execute_activity(
                    compare_products, products, preferences,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                comparison = await compare_products(products, preferences)
            
            if not comparison["recommendation"]:
                return {"success": False, "message": "No good matches found", "products_found": len(products)}
            
            best_product = comparison["recommendation"]["product"]
            
            # Step 3: Check if it's in stock
            if TEMPORAL_AVAILABLE:
                inventory = await workflow.execute_activity(
                    check_inventory, best_product,
                    start_to_close_timeout=timedelta(seconds=30)
                )
            else:
                inventory = await check_inventory(best_product)
            
            if not inventory["in_stock"]:
                return {
                    "success": False,
                    "message": f"{best_product.name} is out of stock",
                    "estimated_restock": inventory["estimated_delivery"],
                    "alternatives": [p.name for p in products[1:3]]
                }
            
            # Step 4: Try to get a deal
            if TEMPORAL_AVAILABLE:
                deal = await workflow.execute_activity(
                    negotiate_deal, best_product, request.budget,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=self.retry_policy
                )
            else:
                deal = await negotiate_deal(best_product, request.budget)
            
            # Step 5: Buy it if urgent or good deal
            if request.urgency == "high" or deal.discount_percent > 10:
                if TEMPORAL_AVAILABLE:
                    purchase = await workflow.execute_activity(
                        process_purchase, deal, request.user_id,
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=self.retry_policy
                    )
                else:
                    purchase = await process_purchase(deal, request.user_id)
                
                return {
                    "success": True,
                    "action": "bought_it",
                    "deal": {
                        "product": deal.product.name,
                        "was": deal.original_price,
                        "now": deal.negotiated_price,
                        "discount": deal.discount_percent,
                        "saved": deal.original_price - deal.negotiated_price
                    },
                    "purchase": purchase,
                    "user_id": request.user_id
                }
            else:
                # Need approval
                return {
                    "success": True,
                    "action": "needs_approval",
                    "deal": {
                        "product": deal.product.name,
                        "was": deal.original_price,
                        "now": deal.negotiated_price,
                        "discount": deal.discount_percent,
                        "saved": deal.original_price - deal.negotiated_price,
                        "terms": deal.terms,
                        "expires": deal.expires_at.isoformat()
                    },
                    "inventory": inventory,
                    "user_id": request.user_id,
                    "message": "Found a deal - want to buy it?"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": request.user_id,
                "message": "Something went wrong"
            }

async def demo_commerce_agent():
    """Try out the commerce agent"""
    
    print("Commerce Agent Demo")
    print("=" * 50)
    
    # Test requests
    requests = [
        CommerceRequest(
            user_id="user_001",
            request="I need a laptop for coding",
            budget=2000.0,
            preferences={"brands": ["Apple", "Dell"], "features": ["16GB RAM", "SSD"]},
            urgency="normal"
        ),
        CommerceRequest(
            user_id="user_002", 
            request="wireless headphones with noise cancellation",
            budget=300.0,
            preferences={"features": ["ANC", "Bluetooth"]},
            urgency="high"
        )
    ]
    
    for i, request in enumerate(requests, 1):
        print(f"\nRequest {i}/{len(requests)}")
        print(f"User: {request.user_id}")
        print(f"Wants: {request.request}")
        print(f"Budget: ${request.budget}")
        print("-" * 30)
        
        workflow = CommerceAgentWorkflow()
        result = await workflow.run(request)
        
        print(f"\nResult:")
        if result["success"]:
            if result["action"] == "bought_it":
                print(f"Bought it!")
                deal = result["deal"]
                print(f"   Product: {deal['product']}")
                print(f"   Price: ${deal['was']:.2f} → ${deal['now']:.2f}")
                print(f"   Saved: ${deal['saved']:.2f} ({deal['discount']:.1f}% off)")
                
                if result["purchase"]["success"]:
                    print(f"   Order: {result['purchase']['order_id']}")
            
            elif result["action"] == "needs_approval":
                print(f"Needs approval:")
                deal = result["deal"]
                print(f"   Product: {deal['product']}")
                print(f"   Price: ${deal['was']:.2f} → ${deal['now']:.2f}")
                print(f"   Saved: ${deal['saved']:.2f} ({deal['discount']:.1f}% off)")
        else:
            print(f"Failed: {result['message']}")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        if i < len(requests):
            print("\n" + "=" * 50)
    
    print(f"\nDone! Processed {len(requests)} requests.")

if __name__ == "__main__":
    asyncio.run(demo_commerce_agent())

