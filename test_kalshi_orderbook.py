#!/usr/bin/env python3
"""Test script to verify Kalshi orderbook API response structure."""
import os
import sys
import json
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import base64
import time

# Load credentials
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY")
KALSHI_API_HOST = os.environ.get("KALSHI_API_HOST", "https://trading-api.kalshi.com/trade-api/v2")

if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY:
    print("Error: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY must be set")
    sys.exit(1)

# Load private key
private_key_str = KALSHI_PRIVATE_KEY.replace("\\n", "\n")
try:
    private_key = serialization.load_pem_private_key(
        private_key_str.encode(),
        password=None,
        backend=default_backend()
    )
except Exception as e:
    print(f"Error loading private key: {e}")
    sys.exit(1)

def make_request(method, path, params=None):
    """Make authenticated Kalshi API request."""
    timestamp = str(int(time.time()))
    body = ""
    
    # Build query string for params
    query_string = ""
    if params:
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            path = f"{path}?{query_string}"
    
    message = f"{timestamp}{method}{path}{body}"
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    signature_b64 = base64.b64encode(signature).decode()
    
    headers = {
        "Authorization": f"{KALSHI_API_KEY_ID}.{signature_b64}.{timestamp}",
        "Content-Type": "application/json"
    }
    
    url = f"{KALSHI_API_HOST}{path}"
    response = requests.request(method, url, headers=headers, json=body if body else None)
    return response

print("="*80)
print("TESTING KALSHI ORDERBOOK API STRUCTURE")
print("="*80)

# Get a live market
print("\n1. Getting a live market ticker...")
try:
    response = make_request("GET", "/markets", params={"limit": 20, "status": "open"})
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    
    result = response.json()
    markets = result.get("markets", [])
    
    if not markets:
        print("No markets found")
        sys.exit(1)
    
    # Find a sports market if possible
    ticker = None
    for market in markets:
        title = market.get("title", "").lower()
        if any(sport in title for sport in ["nba", "nfl", "nhl", "basketball", "football"]):
            ticker = market.get("ticker")
            print(f"Found sports market: {ticker} - {market.get('title')}")
            break
    
    if not ticker:
        ticker = markets[0].get("ticker")
        print(f"Using first market: {ticker} - {markets[0].get('title')}")
    
    print(f"\nUsing ticker: {ticker}\n")
    
except Exception as e:
    print(f"Error getting markets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get orderbook
print("2. Getting orderbook...")
try:
    response = make_request("GET", f"/markets/{ticker}/orderbook")
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    
    result = response.json()
    
    print(f"\nRaw response structure:")
    print(f"Top-level keys: {list(result.keys())}")
    
    # Check if nested
    if "orderbook" in result:
        orderbook = result["orderbook"]
        print(f"\nOrderbook is nested - unwrapping...")
    else:
        orderbook = result
    
    print(f"\nOrderbook keys: {list(orderbook.keys())}")
    
    # Check for yes/no arrays
    yes_raw = orderbook.get("yes", [])
    no_raw = orderbook.get("no", [])
    
    print(f"\nYES array: {len(yes_raw)} items")
    if yes_raw:
        print(f"  Type: {type(yes_raw)}")
        print(f"  First item: {yes_raw[0]} (type: {type(yes_raw[0])})")
        print(f"  Last item: {yes_raw[-1]} (type: {type(yes_raw[-1])})")
        if len(yes_raw) > 1:
            print(f"  Sample items: {yes_raw[:3]}")
    
    print(f"\nNO array: {len(no_raw)} items")
    if no_raw:
        print(f"  Type: {type(no_raw)}")
        print(f"  First item: {no_raw[0]} (type: {type(no_raw[0])})")
        print(f"  Last item: {no_raw[-1]} (type: {type(no_raw[-1])})")
        if len(no_raw) > 1:
            print(f"  Sample items: {no_raw[:3]}")
    
    # Check for other fields
    print(f"\nOther orderbook fields:")
    for key in orderbook.keys():
        if key not in ["yes", "no"]:
            val = orderbook[key]
            if isinstance(val, (list, dict)):
                print(f"  {key}: {type(val)} - {len(val)} items")
            else:
                print(f"  {key}: {type(val)} - {str(val)[:100]}")
    
    # Check for yes_bids/yes_asks (should not exist)
    print(f"\nChecking for incorrect field names:")
    print(f"  yes_bids in orderbook: {'yes_bids' in orderbook}")
    print(f"  yes_asks in orderbook: {'yes_asks' in orderbook}")
    print(f"  no_bids in orderbook: {'no_bids' in orderbook}")
    print(f"  no_asks in orderbook: {'no_asks' in orderbook}")
    
    # Show full structure (truncated)
    print(f"\nFull orderbook structure (first 1500 chars):")
    print(json.dumps(orderbook, indent=2)[:1500])
    
    # Test parsing logic
    print(f"\n" + "="*80)
    print("TESTING PARSING LOGIC")
    print("="*80)
    
    # Parse YES bids
    yes_bids = []
    if yes_raw:
        for order in yes_raw:
            if isinstance(order, list) and len(order) >= 1:
                try:
                    price = int(order[0]) if len(order) > 0 else 0
                    size = int(order[1]) if len(order) > 1 else 0
                    if 0 < price <= 100:
                        yes_bids.append({"price": price, "size": size})
                except (ValueError, TypeError):
                    continue
        yes_bids.sort(key=lambda x: x["price"])
    
    print(f"\nParsed YES bids: {len(yes_bids)}")
    if yes_bids:
        print(f"  Best bid (last element): {yes_bids[-1]}")
        print(f"  Worst bid (first element): {yes_bids[0]}")
        print(f"  All YES bids: {yes_bids[:5]}... (showing first 5)")
    
    # Parse NO bids
    no_bids = []
    if no_raw:
        for order in no_raw:
            if isinstance(order, list) and len(order) >= 1:
                try:
                    price = int(order[0]) if len(order) > 0 else 0
                    size = int(order[1]) if len(order) > 1 else 0
                    if 0 < price <= 100:
                        no_bids.append({"price": price, "size": size})
                except (ValueError, TypeError):
                    continue
        no_bids.sort(key=lambda x: x["price"])
    
    print(f"\nParsed NO bids: {len(no_bids)}")
    if no_bids:
        print(f"  Best bid (last element): {no_bids[-1]}")
        print(f"  Worst bid (first element): {no_bids[0]}")
        print(f"  All NO bids: {no_bids[:5]}... (showing first 5)")
    
    # Calculate asks
    yes_asks = []
    no_asks = []
    
    if no_bids:
        best_no_bid_price = no_bids[-1]["price"]
        best_yes_ask_price = 100 - best_no_bid_price
        best_no_bid_size = no_bids[-1]["size"]
        yes_asks.append({"price": best_yes_ask_price, "size": best_no_bid_size})
        print(f"\nCalculated YES ask: {best_yes_ask_price}¢ (from NO bid: {best_no_bid_price}¢)")
    
    if yes_bids:
        best_yes_bid_price = yes_bids[-1]["price"]
        best_no_ask_price = 100 - best_yes_bid_price
        best_yes_bid_size = yes_bids[-1]["size"]
        no_asks.append({"price": best_no_ask_price, "size": best_yes_bid_size})
        print(f"Calculated NO ask: {best_no_ask_price}¢ (from YES bid: {best_yes_bid_price}¢)")
    
    # Final odds
    if yes_bids and yes_asks:
        yes_bid_price = yes_bids[-1]["price"]
        yes_ask_price = yes_asks[0]["price"]
        yes_mid = (yes_bid_price + yes_ask_price) / 2
        print(f"\n✅ Final YES odds: {yes_mid:.1f}% (Bid: {yes_bid_price}¢, Ask: {yes_ask_price}¢)")
    
    if no_bids and no_asks:
        no_bid_price = no_bids[-1]["price"]
        no_ask_price = no_asks[0]["price"]
        no_mid = (no_bid_price + no_ask_price) / 2
        print(f"❌ Final NO odds: {no_mid:.1f}% (Bid: {no_bid_price}¢, Ask: {no_ask_price}¢)")
    
except Exception as e:
    print(f"Error getting orderbook: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
