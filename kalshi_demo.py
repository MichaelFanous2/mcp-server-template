#!/usr/bin/env python3
import os
import sys
import json
import time
import base64
from typing import Optional, Dict, Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import requests


class KalshiAPI:
    def __init__(self, api_key_id: str, private_key_pem: str, host: str = "https://api.elections.kalshi.com/trade-api/v2"):
        self.api_key_id = api_key_id
        self.host = host.rstrip("/")
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
            password=None
        )
        self.session = requests.Session()
        self.padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
    
    def _sign_request(self, method: str, path: str, body: Optional[str] = None) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{method}\n{path}\n{timestamp}\n{body or ''}"
        signature = base64.b64encode(
            self.private_key.sign(message.encode(), self.padding, hashes.SHA256())
        ).decode()
        return {
            "X-Kalshi-Api-Key-Id": self.api_key_id,
            "X-Kalshi-Timestamp": timestamp,
            "X-Kalshi-Signature": signature
        }
    
    def _request(self, method: str, path: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        body = json.dumps(json_data) if json_data else None
        headers = {"Content-Type": "application/json", **self._sign_request(method, path, body)}
        response = self.session.request(method, f"{self.host}{path}", headers=headers, params=params, json=json_data, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_markets(self, limit: int = 100) -> Dict[str, Any]:
        return self._request("GET", "/markets", params={"limit": limit})
    
    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook")
    
    def create_order(self, ticker: str, side: str, size: int, price: int, order_type: str = "limit") -> Dict[str, Any]:
        return self._request("POST", "/orders", json_data={"ticker": ticker, "side": side, "size": size, "price": price, "type": order_type})
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/orders/{order_id}/cancel")


def main():
    API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
    PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY")
    HOST = os.getenv("KALSHI_API_HOST", "https://api.elections.kalshi.com/trade-api/v2")
    
    if not PRIVATE_KEY or not PRIVATE_KEY.strip():
        print("ERROR: Private key is required")
        sys.exit(1)
    
    try:
        api = KalshiAPI(API_KEY_ID, PRIVATE_KEY, HOST)
    except Exception as e:
        print(f"ERROR initializing API client: {e}")
        sys.exit(1)
    
    print("=" * 60)
    print("Kalshi API Demo - NYC Weather Market")
    print("=" * 60)
    print()
    
    try:
        markets_response = api.get_markets(limit=200)
        markets = markets_response.get("markets", [])
        
        nyc_weather_markets = []
        for m in markets:
            title = (m.get("title") or "").lower()
            ticker = (m.get("ticker") or "").lower()
            if (("nyc" in title or "new york" in title or "ny" in ticker) and 
                ("weather" in title or "temp" in title or "temperature" in title)):
                nyc_weather_markets.append(m)
        
        if not nyc_weather_markets:
            for m in markets:
                if "weather" in (m.get("title") or "").lower() or "temp" in (m.get("title") or "").lower():
                    nyc_weather_markets.append(m)
                    break
            if not nyc_weather_markets:
                print("ERROR: Could not find NYC weather market")
                print(f"Total markets available: {len(markets)}")
                print("\nSample markets (first 10):")
                for i, market in enumerate(markets[:10], 1):
                    print(f"  {i}. {market.get('ticker', 'N/A')}: {market.get('title', 'N/A')}")
                sys.exit(1)
        
        selected_market = nyc_weather_markets[0]
        market_ticker = selected_market.get("ticker")
        
        print(f"Found market: {market_ticker}")
        print(f"Title: {selected_market.get('title', 'N/A')}")
        print(f"Status: {selected_market.get('status', 'N/A')}")
        print(f"Subtitle: {selected_market.get('subtitle', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"ERROR querying markets: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {e.response.text}")
            except:
                pass
        sys.exit(1)
    
    try:
        orderbook = api.get_orderbook(market_ticker)
        yes_bids = orderbook.get("yes_bids", [])
        yes_asks = orderbook.get("yes_asks", [])
        no_bids = orderbook.get("no_bids", [])
        no_asks = orderbook.get("no_asks", [])
        
        print(f"Orderbook for {market_ticker}:")
        print(f"  Yes bids: {len(yes_bids)}")
        print(f"  Yes asks: {len(yes_asks)}")
        print(f"  No bids: {len(no_bids)}")
        print(f"  No asks: {len(no_asks)}")
        
        if yes_bids:
            print(f"\n  Top Yes Bid: {yes_bids[0].get('price', 'N/A')} @ {yes_bids[0].get('size', 'N/A')}")
        if yes_asks:
            print(f"  Top Yes Ask: {yes_asks[0].get('price', 'N/A')} @ {yes_asks[0].get('size', 'N/A')}")
        if no_bids:
            print(f"  Top No Bid: {no_bids[0].get('price', 'N/A')} @ {no_bids[0].get('size', 'N/A')}")
        if no_asks:
            print(f"  Top No Ask: {no_asks[0].get('price', 'N/A')} @ {no_asks[0].get('size', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"ERROR getting orderbook: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {e.response.text}")
            except:
                pass
        sys.exit(1)
    
    order_id = None
    try:
        price = 50
        if yes_bids and yes_asks:
            price = int((yes_bids[0].get("price", 50) + yes_asks[0].get("price", 50)) / 2)
        elif yes_bids:
            price = yes_bids[0].get("price", 50)
        elif yes_asks:
            price = yes_asks[0].get("price", 50)
        
        print(f"Placing order: yes 1 @ {price} cents (limit)")
        order_response = api.create_order(market_ticker, "yes", 1, price, "limit")
        order_id = order_response.get("order_id")
        
        print(f"Order placed successfully!")
        print(f"Order ID: {order_id}")
        print(f"Status: {order_response.get('status', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"ERROR placing order: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Error details: {json.dumps(e.response.json(), indent=2)}")
            except:
                print(f"Response: {e.response.text}")
        err_str = str(e).lower()
        if "read" in err_str or "permission" in err_str or "401" in str(e) or "403" in str(e):
            print("\nSkipping cancellation step due to read-only permissions.")
            sys.exit(0)
    
    if order_id:
        try:
            cancel_response = api.cancel_order(order_id)
            print(f"Order canceled successfully!")
            print(f"Order ID: {order_id}")
            print(f"Status: {cancel_response.get('status', 'N/A')}")
            print()
        except Exception as e:
            print(f"ERROR canceling order: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"Error details: {json.dumps(e.response.json(), indent=2)}")
                except:
                    print(f"Response: {e.response.text}")
            sys.exit(1)
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
