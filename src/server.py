#!/usr/bin/env python3
import os
import time
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import requests

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from apscheduler.schedulers.background import BackgroundScheduler


# =========================
# ENV
# =========================
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]

KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY")
KALSHI_API_HOST = os.environ.get("KALSHI_API_HOST", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_CHECK_INTERVAL = int(os.environ.get("KALSHI_CHECK_INTERVAL", "3600"))
KALSHI_ALERTS_ENABLED = os.environ.get("KALSHI_ALERTS_ENABLED", "true").lower() == "true"
KALSHI_INSIGHTS_ENABLED = os.environ.get("KALSHI_INSIGHTS_ENABLED", "true").lower() == "true"
KALSHI_SCHEDULE_DAYS = os.environ.get("KALSHI_SCHEDULE_DAYS", "sat,sun")
KALSHI_SCHEDULE_HOURS = os.environ.get("KALSHI_SCHEDULE_HOURS", "9-21")

PARALLEL_API_KEY = os.environ.get("PARALLEL_API_KEY")
PARALLEL_MONITOR_INTERVAL = int(os.environ.get("PARALLEL_MONITOR_INTERVAL", "300"))
PARALLEL_MONITOR_ENABLED = os.environ.get("PARALLEL_MONITOR_ENABLED", "true").lower() == "true"


# =========================
# CLIENTS / SERVER
# =========================
twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")

class KalshiAPI:
    """Read-only Kalshi API client. Trading operations are explicitly disabled."""
    def __init__(self, api_key_id: str, private_key_pem: str, host: str):
        self.api_key_id = api_key_id
        self.host = host.rstrip("/")
        
        if isinstance(private_key_pem, str):
            private_key_pem = private_key_pem.replace("\\n", "\n")
            if not private_key_pem.startswith("-----BEGIN"):
                raise ValueError("Private key must be in PEM format starting with -----BEGIN RSA PRIVATE KEY-----")
        
        try:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
                password=None
            )
        except Exception as e:
            raise ValueError(f"Failed to load private key: {str(e)}. Make sure the key is properly formatted with newlines.")
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
    
    def _rate_limit(self):
        global KALSHI_RATE_LIMIT_QUEUE
        now_ts = time.time()
        KALSHI_RATE_LIMIT_QUEUE = [t for t in KALSHI_RATE_LIMIT_QUEUE if now_ts - t < 1.0]
        
        if len(KALSHI_RATE_LIMIT_QUEUE) >= KALSHI_RATE_LIMIT:
            sleep_time = 1.0 - (now_ts - KALSHI_RATE_LIMIT_QUEUE[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            KALSHI_RATE_LIMIT_QUEUE = [t for t in KALSHI_RATE_LIMIT_QUEUE if now_ts - t < 1.0]
        
        KALSHI_RATE_LIMIT_QUEUE.append(time.time())
    
    def _request(self, method: str, path: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        if method.upper() != "GET":
            raise ValueError("Kalshi API is read-only. Only GET requests are allowed.")
        
        if "/orders" in path or "/portfolio" in path or "/exchange/balance" in path:
            raise ValueError("Trading operations are not allowed. This API is read-only.")
        
        self._rate_limit()
        
        body = json.dumps(json_data) if json_data else None
        headers = {"Content-Type": "application/json", **self._sign_request(method, path, body)}
        logger.info(f"[API CALL] Kalshi API - {method} {path}, params: {params}")
        response = self.session.request(method, f"{self.host}{path}", headers=headers, params=params, json=json_data, timeout=30)
        response.raise_for_status()
        logger.info(f"[API CALL] Kalshi API response - {method} {path} - Status: {response.status_code}")
        return response.json()
    
    def get_markets(self, limit: int = 100, **kwargs) -> Dict[str, Any]:
        params = {"limit": limit, **kwargs}
        return self._request("GET", "/markets", params=params)
    
    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook")
    
    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}")

kalshi_api = None
if KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY:
    try:
        kalshi_api = KalshiAPI(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY, KALSHI_API_HOST)
    except Exception as e:
        print(f"Warning: Failed to initialize Kalshi API: {e}")

scheduler = BackgroundScheduler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# STATE
# =========================
AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CALL_CONTEXT: Dict[str, Dict[str, Any]] = {}
MAX_CALL_DURATION_SECONDS = 240

KALSHI_WATCHES_FILE = Path("/tmp/kalshi_watches.json")
KALSHI_WATCHES: Dict[str, Dict[str, Any]] = {}
KALSHI_INSIGHTS: List[Dict[str, Any]] = []
KALSHI_PRICE_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
KALSHI_VOLUME_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
KALSHI_RATE_LIMIT_QUEUE: List[float] = []
KALSHI_RATE_LIMIT = 20

PARALLEL_MONITORS: Dict[str, Dict[str, Any]] = {}
PARALLEL_MONITORS_FILE = Path("/tmp/parallel_monitors.json")
PARALLEL_ALERTS: List[Dict[str, Any]] = []


# =========================
# HELPERS
# =========================
def now() -> float:
    return time.time()


def audio_path(key: str) -> Path:
    return AUDIO_DIR / f"{key}.mp3"


def write_audio(key: str, audio: bytes):
    p = audio_path(key)
    with open(p, "wb") as f:
        f.write(audio)


def greeting_for_topic(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return "Hey. Go ahead."
    return f"Hey. Let's talk about {topic}. Go ahead."


def load_kalshi_watches():
    global KALSHI_WATCHES
    if KALSHI_WATCHES_FILE.exists():
        try:
            with open(KALSHI_WATCHES_FILE, "r") as f:
                KALSHI_WATCHES = json.load(f)
        except:
            KALSHI_WATCHES = {}


def save_kalshi_watches():
    try:
        with open(KALSHI_WATCHES_FILE, "w") as f:
            json.dump(KALSHI_WATCHES, f)
    except:
        pass


def generate_market_insights(ticker: str, market_data: Dict[str, Any], orderbook: Dict[str, Any]) -> str:
    if not KALSHI_INSIGHTS_ENABLED or not OPENAI_API_KEY:
        return ""
    
    yes_bids = orderbook.get("yes_bids", [])
    yes_asks = orderbook.get("yes_asks", [])
    mid_price = None
    if yes_bids and yes_asks:
        mid_price = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
    elif yes_bids:
        mid_price = yes_bids[0].get("price", 0)
    elif yes_asks:
        mid_price = yes_asks[0].get("price", 100)
    
    market_info = f"""
Market: {market_data.get('title', 'N/A')}
Ticker: {ticker}
Status: {market_data.get('status', 'N/A')}
Current Mid Price: {mid_price} cents
Yes Bids: {len(yes_bids)}
Yes Asks: {len(yes_asks)}
"""
    
    payload = {
        "model": "gpt-4.1-mini",
        "instructions": (
            "You are a betting market analyst. Provide a brief, actionable insight (2-3 sentences max) "
            "on whether this market presents a good betting opportunity. Consider price, liquidity, "
            "and market dynamics. Be direct and practical."
        ),
        "input": f"Analyze this Kalshi prediction market:\n{market_info}\n\nProvide betting insight:",
        "max_output_tokens": 150,
        "temperature": 0.7,
    }
    
    try:
        logger.info(f"[API CALL] OpenAI generate_market_insights - ticker: '{ticker}', mid_price: {mid_price}")
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["output"][0]["content"][0]["text"].strip()
    except:
        return ""


def check_kalshi_watches():
    if not kalshi_api or not KALSHI_ALERTS_ENABLED:
        return
    
    for ticker, watch in list(KALSHI_WATCHES.items()):
        try:
            orderbook = kalshi_api.get_orderbook(ticker)
            market_data = kalshi_api.get_market(ticker)
            
            yes_bids = orderbook.get("yes_bids", [])
            yes_asks = orderbook.get("yes_asks", [])
            
            current_price = None
            if yes_bids and yes_asks:
                current_price = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
            elif yes_bids:
                current_price = yes_bids[0].get("price", 0)
            elif yes_asks:
                current_price = yes_asks[0].get("price", 100)
            
            alert_price = watch.get("alert_price")
            direction = watch.get("direction", "above")
            
            triggered = False
            if alert_price and current_price:
                if direction == "above" and current_price >= alert_price:
                    triggered = True
                elif direction == "below" and current_price <= alert_price:
                    triggered = True
            
            if triggered:
                volume_data = analyze_orderbook_volume_distribution(orderbook)
                risk_scores = calculate_risk_scores(ticker, market_data, orderbook, volume_data)
                volume_shifts = analyze_volume_shifts(ticker)
                
                price_history = KALSHI_PRICE_HISTORY.get(ticker, [])
                price_change_1h = None
                if len(price_history) >= 2:
                    recent_prices = [p.get("mid_price") for p in price_history[-5:] if p.get("mid_price")]
                    if len(recent_prices) >= 2:
                        price_change_1h = ((current_price - recent_prices[0]) / recent_prices[0] * 100) if recent_prices[0] > 0 else 0
                
                insights = generate_market_insights(ticker, market_data, orderbook) if KALSHI_INSIGHTS_ENABLED else ""
                
                context_parts = []
                if price_change_1h:
                    context_parts.append(f"Price moved {abs(price_change_1h):.1f}% in recent checks")
                
                if "error" not in volume_shifts:
                    if volume_shifts['trend'] == "increasing":
                        context_parts.append(f"Volume up {volume_shifts['volume_change_pct']:.1f}%")
                    elif volume_shifts['trend'] == "decreasing":
                        context_parts.append(f"Volume down {abs(volume_shifts['volume_change_pct']):.1f}%")
                
                spread = risk_scores.get("spread", 0)
                if spread < 2:
                    context_parts.append("Tight spread - good liquidity")
                elif spread > 10:
                    context_parts.append("Wide spread - low liquidity")
                
                if volume_data['max_bid_concentration'] > 70 or volume_data['max_ask_concentration'] > 70:
                    context_parts.append(f"⚠️ High concentration risk ({max(volume_data['max_bid_concentration'], volume_data['max_ask_concentration']):.0f}%)")
                
                context_text = " | ".join(context_parts) if context_parts else "No significant context"
                insight_text = f"\n\n💡 AI Insight: {insights}" if insights else ""
                
                recommendation = ""
                if risk_scores.get('overall_risk', 100) < 30 and spread < 3:
                    recommendation = "\n✅ Recommendation: Low risk, good liquidity - strong opportunity"
                elif risk_scores.get('overall_risk', 0) > 70:
                    recommendation = "\n⚠️ Recommendation: High risk - proceed with caution"
                elif volume_data['max_bid_concentration'] > 70 or volume_data['max_ask_concentration'] > 70:
                    recommendation = "\n⚠️ Recommendation: High concentration risk - market may be manipulated"
                
                alert_msg = (
                    f"🚨 Kalshi Alert: {ticker} ({market_data.get('title', 'N/A')})\n"
                    f"Price: {current_price:.2f} cents (target: {alert_price} {direction})\n"
                    f"Context: {context_text}{insight_text}{recommendation}"
                )
                
                KALSHI_INSIGHTS.append({
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "price": current_price,
                    "alert_price": alert_price,
                    "insight": insights,
                    "risk_scores": risk_scores,
                    "volume_data": volume_data,
                    "message": alert_msg
                })
                
                print(f"[KALSHI ALERT] {alert_msg}")
                
                if watch.get("remove_after_trigger", False):
                    del KALSHI_WATCHES[ticker]
                    save_kalshi_watches()
        except Exception as e:
            print(f"[KALSHI ERROR] Failed to check {ticker}: {e}")


def analyze_orderbook_volume_distribution(orderbook: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze volume distribution and concentration risk."""
    yes_bids = orderbook.get("yes_bids", [])
    yes_asks = orderbook.get("yes_asks", [])
    no_bids = orderbook.get("no_bids", [])
    no_asks = orderbook.get("no_asks", [])
    
    total_yes_bid_volume = sum(b.get("size", 0) for b in yes_bids)
    total_yes_ask_volume = sum(a.get("size", 0) for a in yes_asks)
    total_no_bid_volume = sum(b.get("size", 0) for b in no_bids)
    total_no_ask_volume = sum(a.get("size", 0) for a in no_asks)
    
    total_volume = total_yes_bid_volume + total_yes_ask_volume + total_no_bid_volume + total_no_ask_volume
    
    yes_bid_distribution = []
    yes_ask_distribution = []
    
    for bid in yes_bids[:10]:
        size = bid.get("size", 0)
        if total_yes_bid_volume > 0:
            pct = (size / total_yes_bid_volume) * 100
            yes_bid_distribution.append({"size": size, "price": bid.get("price"), "pct": pct})
    
    for ask in yes_asks[:10]:
        size = ask.get("size", 0)
        if total_yes_ask_volume > 0:
            pct = (size / total_yes_ask_volume) * 100
            yes_ask_distribution.append({"size": size, "price": ask.get("price"), "pct": pct})
    
    max_bid_concentration = max([d["pct"] for d in yes_bid_distribution], default=0)
    max_ask_concentration = max([d["pct"] for d in yes_ask_distribution], default=0)
    
    return {
        "total_volume": total_volume,
        "yes_bid_volume": total_yes_bid_volume,
        "yes_ask_volume": total_yes_ask_volume,
        "no_bid_volume": total_no_bid_volume,
        "no_ask_volume": total_no_ask_volume,
        "yes_bid_distribution": yes_bid_distribution,
        "yes_ask_distribution": yes_ask_distribution,
        "max_bid_concentration": max_bid_concentration,
        "max_ask_concentration": max_ask_concentration,
        "buyer_seller_ratio": total_yes_bid_volume / total_yes_ask_volume if total_yes_ask_volume > 0 else None
    }


def calculate_risk_scores(ticker: str, market_data: Dict[str, Any], orderbook: Dict[str, Any], volume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive risk scores."""
    yes_bids = orderbook.get("yes_bids", [])
    yes_asks = orderbook.get("yes_asks", [])
    
    if not yes_bids or not yes_asks:
        return {"error": "Insufficient data for risk analysis"}
    
    bid_price = yes_bids[0].get("price", 0)
    ask_price = yes_asks[0].get("price", 100)
    mid_price = (bid_price + ask_price) / 2
    spread = ask_price - bid_price
    
    price_history = KALSHI_PRICE_HISTORY.get(ticker, [])
    volume_history = KALSHI_VOLUME_HISTORY.get(ticker, [])
    
    volatility_score = 0
    if len(price_history) >= 5:
        prices = [p.get("mid_price") for p in price_history[-10:] if p.get("mid_price")]
        if len(prices) >= 5:
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
            volatility_score = min(100, (avg_change / mid_price * 100) * 10) if mid_price > 0 else 0
    
    liquidity_score = 100
    if spread > 5:
        liquidity_score -= (spread - 5) * 5
    if volume_data["total_volume"] < 100:
        liquidity_score -= 20
    liquidity_score = max(0, min(100, liquidity_score))
    
    concentration_risk = 0
    if volume_data["max_bid_concentration"] > 50:
        concentration_risk += (volume_data["max_bid_concentration"] - 50) * 2
    if volume_data["max_ask_concentration"] > 50:
        concentration_risk += (volume_data["max_ask_concentration"] - 50) * 2
    concentration_risk = min(100, concentration_risk)
    
    volume_stability = 100
    if len(volume_history) >= 3:
        volumes = [v.get("total_volume", 0) for v in volume_history[-5:]]
        if len(volumes) >= 3:
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volume_data["total_volume"]
            if avg_volume > 0:
                volume_change_pct = abs((current_volume - avg_volume) / avg_volume * 100)
                if volume_change_pct > 50:
                    volume_stability = max(0, 100 - volume_change_pct)
    
    overall_risk = (volatility_score * 0.3 + (100 - liquidity_score) * 0.3 + concentration_risk * 0.2 + (100 - volume_stability) * 0.2)
    
    return {
        "volatility_score": round(volatility_score, 1),
        "liquidity_score": round(liquidity_score, 1),
        "concentration_risk": round(concentration_risk, 1),
        "volume_stability": round(volume_stability, 1),
        "overall_risk": round(overall_risk, 1),
        "spread": spread,
        "total_volume": volume_data["total_volume"]
    }


def analyze_volume_shifts(ticker: str) -> Dict[str, Any]:
    """Analyze volume shifts over time."""
    volume_history = KALSHI_VOLUME_HISTORY.get(ticker, [])
    
    if len(volume_history) < 2:
        return {"error": "Insufficient volume history"}
    
    recent = volume_history[-5:]
    older = volume_history[-10:-5] if len(volume_history) >= 10 else volume_history[:len(volume_history)-5]
    
    recent_avg = sum(v.get("total_volume", 0) for v in recent) / len(recent) if recent else 0
    older_avg = sum(v.get("total_volume", 0) for v in older) / len(older) if older else 0
    
    volume_change_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
    
    current = volume_history[-1] if volume_history else {}
    current_volume = current.get("total_volume", 0)
    
    trend = "stable"
    if volume_change_pct > 20:
        trend = "increasing"
    elif volume_change_pct < -20:
        trend = "decreasing"
    
    return {
        "recent_avg_volume": round(recent_avg, 0),
        "older_avg_volume": round(older_avg, 0),
        "volume_change_pct": round(volume_change_pct, 1),
        "current_volume": current_volume,
        "trend": trend,
        "data_points": len(volume_history)
    }


load_kalshi_watches()


class ParallelAPI:
    """Parallel AI Extract and Search API client."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.parallel.ai/v1beta"
        self.session = requests.Session()
    
    def extract(self, urls: List[str], objective: str, excerpts: bool = True, full_content: bool = True) -> Dict[str, Any]:
        """Extract content from URLs using Parallel API."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "parallel-beta": "search-extract-2025-10-10"
        }
        
        payload = {
            "urls": urls,
            "objective": objective,
            "excerpts": excerpts,
            "full_content": full_content
        }
        
        logger.info(f"[API CALL] Parallel Extract - URLs: {len(urls)}, objective: '{objective[:50]}...'")
        response = self.session.post(
            f"{self.base_url}/extract",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        logger.info(f"[API CALL] Parallel Extract response - Status: {response.status_code}")
        return response.json()
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using Parallel API."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "parallel-beta": "search-extract-2025-10-10"
        }
        
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        logger.info(f"[API CALL] Parallel Search - query: '{query[:50]}...', max_results: {max_results}")
        response = self.session.post(
            f"{self.base_url}/search",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        logger.info(f"[API CALL] Parallel Search response - Status: {response.status_code}")
        return response.json()

parallel_api = None
if PARALLEL_API_KEY:
    try:
        parallel_api = ParallelAPI(PARALLEL_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Parallel API: {e}")


def load_parallel_monitors():
    global PARALLEL_MONITORS
    if PARALLEL_MONITORS_FILE.exists():
        try:
            with open(PARALLEL_MONITORS_FILE, "r") as f:
                PARALLEL_MONITORS = json.load(f)
        except:
            PARALLEL_MONITORS = {}


def save_parallel_monitors():
    try:
        with open(PARALLEL_MONITORS_FILE, "w") as f:
            json.dump(PARALLEL_MONITORS, f)
    except:
        pass


def format_kalshi_odds(orderbook: Dict[str, Any], ticker: str) -> str:
    """Format Kalshi orderbook data into readable odds/probabilities with bid/ask prices and volumes."""
    yes_bids = orderbook.get("yes_bids", [])
    yes_asks = orderbook.get("yes_asks", [])
    no_bids = orderbook.get("no_bids", [])
    no_asks = orderbook.get("no_asks", [])
    
    if not yes_bids or not yes_asks:
        return f"    💰 Kalshi Market: No active orderbook (ticker: {ticker})\n"
    
    yes_bid_price = yes_bids[0].get("price", 0)
    yes_ask_price = yes_asks[0].get("price", 100)
    yes_mid = (yes_bid_price + yes_ask_price) / 2
    yes_spread = yes_ask_price - yes_bid_price
    yes_bid_vol = yes_bids[0].get("size", 0)
    yes_ask_vol = yes_asks[0].get("size", 0)
    
    result = f"    💰 Kalshi Live Odds (ticker: {ticker}):\n"
    result += f"      ✅ YES: {yes_mid:.1f}% (bid: {yes_bid_price}¢ @ {yes_bid_vol} contracts, ask: {yes_ask_price}¢ @ {yes_ask_vol} contracts, spread: {yes_spread:.1f}¢)\n"
    
    if no_bids and no_asks:
        no_bid_price = no_bids[0].get("price", 0)
        no_ask_price = no_asks[0].get("price", 100)
        no_mid = (no_bid_price + no_ask_price) / 2
        no_spread = no_ask_price - no_bid_price
        no_bid_vol = no_bids[0].get("size", 0)
        no_ask_vol = no_asks[0].get("size", 0)
        result += f"      ❌ NO: {no_mid:.1f}% (bid: {no_bid_price}¢ @ {no_bid_vol} contracts, ask: {no_ask_price}¢ @ {no_ask_vol} contracts, spread: {no_spread:.1f}¢)\n"
    else:
        # Calculate No from Yes (should be ~100 - Yes)
        no_prob = 100 - yes_mid
        result += f"      ❌ NO: ~{no_prob:.1f}% (derived from Yes price)\n"
    
    return result


def parse_kalshi_calendar_content(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse Kalshi calendar page to extract live games, upcoming games, scores, and probabilities."""
    result = {
        "live_games": [],
        "upcoming_games": []
    }
    
    lines = content.split("\n")
    current_game = None
    game_status = None  # "LIVE", "UPCOMING", or None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line == "* * *" or "© 2025" in line:
            continue
        
        # Detect LIVE games
        if "LIVE" in line and ("Q" in line or "HALFTIME" in line or "OT" in line):
            if current_game and current_game.get("teams"):
                if game_status == "LIVE":
                    result["live_games"].append(current_game)
                elif game_status == "UPCOMING":
                    result["upcoming_games"].append(current_game)
            current_game = {"status": "LIVE", "raw_line": line}
            game_status = "LIVE"
        
        # Detect upcoming games (has time like "Dec 27 @ 4:30pm EST" or "@ 7pm")
        if "@" in line and ("pm" in line.lower() or "am" in line.lower() or "EST" in line or "PST" in line):
            if current_game and current_game.get("teams") and game_status != "LIVE":
                result["upcoming_games"].append(current_game)
            if not current_game or current_game.get("status") != "LIVE":
                current_game = {"status": "UPCOMING", "raw_line": line}
                game_status = "UPCOMING"
                # Extract time
                time_match = line
                if "@" in time_match:
                    time_part = time_match.split("@")[-1].strip()
                    current_game["start_time"] = time_part
        
        if current_game:
            # Extract sport type and teams
            if any(x in line for x in ["Pro Football", "College Football", "Pro Basketball", "College Basketball", "Basketball", "Football"]):
                parts = line.split()
                sport_type = None
                teams = []
                
                for j, part in enumerate(parts):
                    if part in ["Pro", "College"] and j + 1 < len(parts):
                        sport_type = f"{part} {parts[j+1]}"
                    elif part.isupper() and len(part) > 1 and part not in ["LIVE", "Q", "HALFTIME", "OT", "EST", "PST", "AM", "PM"]:
                        if len(teams) < 2:
                            teams.append(part)
                
                if sport_type:
                    current_game["sport"] = sport_type
                if teams:
                    current_game["teams"] = teams
            
            # Extract scores (look for patterns like "9 8 7 6 5 4 3 2 1 0" or "78% 22%")
            # Scores often appear as numbers separated by spaces before percentages
            if any(char.isdigit() for char in line) and ("%" in line or len([x for x in line.split() if x.isdigit()]) >= 2):
                # Try to extract score from patterns like "Team1 Score1 Team2 Score2" or number sequences
                parts = line.split()
                numbers = [p for p in parts if p.isdigit() and len(p) <= 3]
                if len(numbers) >= 2 and current_game.get("teams") and len(current_game["teams"]) >= 2:
                    # Likely scores - take first two significant numbers
                    try:
                        score1 = int(numbers[0])
                        score2 = int(numbers[1])
                        if score1 < 200 and score2 < 200:  # Reasonable score range
                            current_game["score"] = f"{current_game['teams'][0]} {score1} - {current_game['teams'][1]} {score2}"
                    except:
                        pass
            
            # Extract probabilities
            if "%" in line:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if "%" in part:
                            pct = int(part.replace("%", "").replace("¢", ""))
                            if j > 0:
                                prev = parts[j-1].lower()
                                if prev in ["yes", "no"]:
                                    current_game[f"{prev}_probability"] = pct
                                # Also check if it's a team probability (like "78% 22%")
                                elif pct > 0 and pct < 100 and not current_game.get("yes_probability"):
                                    # Might be team win probability
                                    if not current_game.get("team1_probability"):
                                        current_game["team1_probability"] = pct
                                    elif not current_game.get("team2_probability"):
                                        current_game["team2_probability"] = pct
                except:
                    pass
            
            # Extract quarter/time info
            if "Q" in line or "HALFTIME" in line or "OT" in line:
                if "HALFTIME" in line:
                    current_game["quarter"] = "HALFTIME"
                elif "OT" in line:
                    current_game["quarter"] = "OT"
                elif "Q" in line:
                    q_parts = line.split("Q")
                    if len(q_parts) > 1:
                        q_num = q_parts[0].split()[-1] if q_parts[0].split() else "?"
                        time_part = q_parts[1].strip() if len(q_parts) > 1 else ""
                        current_game["quarter"] = f"Q{q_num}"
                        if time_part:
                            current_game["time"] = time_part
    
    # Add final game
    if current_game and current_game.get("teams"):
        if game_status == "LIVE":
            result["live_games"].append(current_game)
        elif game_status == "UPCOMING":
            result["upcoming_games"].append(current_game)
    
    return result


def generate_alpha_insight_with_llm(game_info: Dict[str, Any], market_data: Dict[str, Any], orderbook: Dict[str, Any], use_search: bool = True) -> str:
    """Use OpenAI to generate intelligent alpha insights comparing live game state vs market. Optionally uses Parallel search for context."""
    if not KALSHI_INSIGHTS_ENABLED or not OPENAI_API_KEY:
        return ""
    
    yes_bids = orderbook.get("yes_bids", [])
    yes_asks = orderbook.get("yes_asks", [])
    
    if not yes_bids or not yes_asks:
        return ""
    
    market_yes_prob = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
    game_yes_prob = game_info.get("yes_probability")
    
    if not game_yes_prob:
        return ""
    
    spread = yes_asks[0].get("price", 100) - yes_bids[0].get("price", 0)
    bid_depth = sum(b.get("size", 0) for b in yes_bids[:5])
    ask_depth = sum(a.get("size", 0) for a in yes_asks[:5])
    
    teams_str = ", ".join(game_info.get('teams', []))
    sport = game_info.get('sport', 'Unknown')
    quarter = game_info.get('quarter', 'Unknown')
    
    # Optional: Use Parallel search for additional context
    search_context = ""
    if use_search and parallel_api and abs(market_yes_prob - game_yes_prob) > 5:
        try:
            search_query = f"{sport} {teams_str} {quarter} live score"
            logger.info(f"[PARALLEL] Searching for game context: '{search_query}'")
            search_results = parallel_api.search(search_query, max_results=3)
            
            if "results" in search_results and search_results["results"]:
                search_context = "\n\nAdditional Context from Web Search (Third-Party Source):\n"
                search_context += "⚠️ IMPORTANT: This is information from a third-party web search API. "
                search_context += "It may or may not be accurate. Treat it as unbiased context, not verified truth. "
                search_context += "Evaluate whether you agree or disagree with this information, then provide your analysis.\n\n"
                
                for i, result in enumerate(search_results["results"][:3], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", result.get("excerpt", "No content"))
                    url = result.get("url", "")
                    search_context += f"{i}. {title}\n"
                    search_context += f"   {snippet[:200]}...\n"
                    if url:
                        search_context += f"   Source: {url}\n"
                    search_context += "\n"
        except Exception as e:
            logger.warning(f"[PARALLEL] Failed to get search context: {e}")
            search_context = ""
    
    context = f"""
Live Game State (from Kalshi calendar):
- Teams: {teams_str}
- Sport: {sport}
- Quarter/Status: {quarter}
- Calendar Probability: {game_yes_prob}% Yes

Market Data (from Kalshi API):
- Market: {market_data.get('title', 'N/A')}
- Market Price: {market_yes_prob:.1f}% Yes
- Spread: {spread:.1f} cents
- Bid Depth (top 5): {bid_depth} contracts
- Ask Depth (top 5): {ask_depth} contracts

Difference: {abs(market_yes_prob - game_yes_prob):.1f} percentage points
{search_context}
"""
    
    instructions = (
        "You are a sports betting analyst. Analyze the discrepancy between the live game state "
        "and the market price. Consider: game momentum (which quarter, score if available), "
        "market liquidity (spread, depth), and whether the market is mispricing based on current game state. "
    )
    
    if search_context:
        instructions += (
            "\nIMPORTANT: If web search results are provided above, they come from a third-party search API. "
            "This information may or may not be true. Do not blindly accept it as fact. "
            "Treat it as unbiased context - evaluate whether you agree or disagree with the information, "
            "then provide your analysis based on all available information. Be clear about what you're using "
            "from the search results vs. what you're inferring from the market data."
        )
    
    instructions += (
        "\nProvide a brief, actionable insight (2-3 sentences max) on whether this represents alpha. "
        "Be specific about why the market might be wrong."
    )
    
    payload = {
        "model": "gpt-4.1-mini",
        "instructions": instructions,
        "input": f"Analyze this betting opportunity:\n{context}\n\nIs there alpha here? Explain:",
        "max_output_tokens": 200,
        "temperature": 0.7,
    }
    
    try:
        logger.info(f"[API CALL] OpenAI generate_alpha_insight - game: {teams_str}, market_prob: {market_yes_prob:.1f}%, game_prob: {game_yes_prob}%, use_search: {use_search}")
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        insight = data["output"][0]["content"][0]["text"].strip()
        return f"💡 ALPHA: {insight}"
    except Exception as e:
        logger.error(f"[API CALL] Failed to generate LLM alpha insight: {e}")
        return ""


def check_parallel_monitors():
    """Check all Parallel monitors for updates."""
    logger.info(f"[PARALLEL] check_parallel_monitors called - {len(PARALLEL_MONITORS)} active monitors")
    if not parallel_api or not PARALLEL_MONITOR_ENABLED:
        logger.info("[PARALLEL] Monitoring disabled or API not configured")
        return
    
    for monitor_id, monitor in list(PARALLEL_MONITORS.items()):
        try:
            urls = monitor.get("urls", [])
            objective = monitor.get("objective", "")
            
            if not urls:
                logger.warning(f"[PARALLEL] Monitor {monitor_id} has no URLs")
                continue
            
            logger.info(f"[PARALLEL] Checking monitor '{monitor.get('name', monitor_id)}' - {len(urls)} URL(s)")
            result = parallel_api.extract(urls, objective, excerpts=True, full_content=True)
            
            if "results" in result and result["results"]:
                content = result["results"][0].get("full_content", "")
                previous_content = monitor.get("last_content", "")
                
                if content != previous_content or monitor.get("check_count", 0) == 0:
                    monitor["last_content"] = content
                    monitor["last_check"] = datetime.now().isoformat()
                    monitor["check_count"] = monitor.get("check_count", 0) + 1
                    save_parallel_monitors()
                    
                    alert_msg = f"🔄 Parallel Monitor: {monitor.get('name', monitor_id)}\n"
                    
                    if monitor.get("type") == "kalshi_calendar":
                        games_data = parse_kalshi_calendar_content(content)
                        live_games = games_data.get("live_games", [])
                        upcoming_games = games_data.get("upcoming_games", [])
                        
                        # Format LIVE GAMES with scores and probabilities
                        if live_games:
                            alert_msg += f"\n🔥 LIVE GAMES ({len(live_games)}):\n\n"
                            
                            for game in live_games[:15]:
                                teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                                sport = game.get("sport", "Unknown Sport")
                                quarter = game.get("quarter", "Unknown")
                                
                                alert_msg += f"  {sport}: {teams_str}\n"
                                
                                # SCORE (most important for live games)
                                if game.get("score"):
                                    alert_msg += f"    📊 Score: {game['score']}\n"
                                else:
                                    alert_msg += f"    📊 Score: Not available\n"
                                
                                # STATUS
                                alert_msg += f"    ⏱️  Status: {quarter}"
                                if game.get("time"):
                                    alert_msg += f" - {game['time']}"
                                alert_msg += "\n"
                                
                                # PROBABILITIES - Calendar probabilities
                                calendar_probs = []
                                if game.get("yes_probability"):
                                    calendar_probs.append(f"Yes: {game['yes_probability']}%")
                                if game.get("no_probability"):
                                    calendar_probs.append(f"No: {game['no_probability']}%")
                                if game.get("team1_probability") and game.get("team2_probability"):
                                    calendar_probs.append(f"{game['teams'][0]}: {game['team1_probability']}% | {game['teams'][1]}: {game['team2_probability']}%")
                                
                                if calendar_probs:
                                    alert_msg += f"    📈 Calendar Probabilities: {', '.join(calendar_probs)}\n"
                                
                                # PROBABILITIES - Kalshi API market prices
                                if game.get("teams") and kalshi_api:
                                    try:
                                        markets_resp = kalshi_api.get_markets(limit=50)
                                        markets = markets_resp.get("markets", [])
                                        markets = filter_active_markets(markets)
                                        
                                        matching_markets = [
                                            m for m in markets
                                            if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                        ]
                                        
                                        if matching_markets:
                                            market = matching_markets[0]
                                            ticker = market.get("ticker")
                                            orderbook = kalshi_api.get_orderbook(ticker)
                                            
                                            yes_bids = orderbook.get("yes_bids", [])
                                            yes_asks = orderbook.get("yes_asks", [])
                                            if yes_bids and yes_asks:
                                                market_prob = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
                                                spread = yes_asks[0].get("price", 100) - yes_bids[0].get("price", 0)
                                                alert_msg += format_kalshi_odds(orderbook, ticker)
                                                
                                                # Alpha detection
                                                alpha = generate_alpha_insight_with_llm(game, market, orderbook, use_search=True)
                                                if alpha:
                                                    alert_msg += f"    {alpha}\n"
                                        else:
                                            alert_msg += f"    💰 Kalshi Market: No matching market found\n"
                                    except Exception as e:
                                        logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                        alert_msg += f"    💰 Kalshi Market: Error fetching data\n"
                                
                                alert_msg += "\n"
                        
                        # Format UPCOMING GAMES with times and probabilities
                        if upcoming_games:
                            alert_msg += f"\n⏰ UPCOMING GAMES ({len(upcoming_games)}):\n\n"
                            
                            for game in upcoming_games[:15]:
                                teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                                sport = game.get("sport", "Unknown Sport")
                                start_time = game.get("start_time", "Time TBD")
                                
                                alert_msg += f"  {sport}: {teams_str}\n"
                                alert_msg += f"    ⏰ Start Time: {start_time}\n"
                                
                                # PROBABILITIES - Calendar probabilities
                                calendar_probs = []
                                if game.get("yes_probability"):
                                    calendar_probs.append(f"Yes: {game['yes_probability']}%")
                                if game.get("no_probability"):
                                    calendar_probs.append(f"No: {game['no_probability']}%")
                                if game.get("team1_probability") and game.get("team2_probability"):
                                    calendar_probs.append(f"{game['teams'][0]}: {game['team1_probability']}% | {game['teams'][1]}: {game['team2_probability']}%")
                                
                                if calendar_probs:
                                    alert_msg += f"    📈 Calendar Probabilities: {', '.join(calendar_probs)}\n"
                                
                                # PROBABILITIES - Kalshi API market prices
                                if game.get("teams") and kalshi_api:
                                    try:
                                        markets_resp = kalshi_api.get_markets(limit=50)
                                        markets = markets_resp.get("markets", [])
                                        markets = filter_active_markets(markets)
                                        
                                        matching_markets = [
                                            m for m in markets
                                            if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                        ]
                                        
                                        if matching_markets:
                                            market = matching_markets[0]
                                            ticker = market.get("ticker")
                                            orderbook = kalshi_api.get_orderbook(ticker)
                                            
                                            yes_bids = orderbook.get("yes_bids", [])
                                            yes_asks = orderbook.get("yes_asks", [])
                                            if yes_bids and yes_asks:
                                                market_prob = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
                                                spread = yes_asks[0].get("price", 100) - yes_bids[0].get("price", 0)
                                                alert_msg += format_kalshi_odds(orderbook, ticker)
                                        else:
                                            alert_msg += f"    💰 Kalshi Market: No matching market found\n"
                                    except Exception as e:
                                        logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                        alert_msg += f"    💰 Kalshi Market: Error fetching data\n"
                                
                                alert_msg += "\n"
                        
                        if not live_games and not upcoming_games:
                            alert_msg += "\n⚠️ No games found in calendar data\n"
                    
                    PARALLEL_ALERTS.append({
                        "monitor_id": monitor_id,
                        "timestamp": datetime.now().isoformat(),
                        "message": alert_msg,
                        "content_preview": content[:500]
                    })
                    
                    logger.info(f"[PARALLEL ALERT] Monitor '{monitor.get('name', monitor_id)}' triggered alert")
                    print(f"[PARALLEL ALERT] {alert_msg}")
        except Exception as e:
            logger.error(f"[PARALLEL ERROR] Failed to check monitor {monitor_id}: {e}")
            print(f"[PARALLEL ERROR] Failed to check monitor {monitor_id}: {e}")


load_parallel_monitors()


def check_single_parallel_monitor(monitor_id: str):
    """Check a single Parallel monitor. Used for per-monitor scheduling."""
    if monitor_id not in PARALLEL_MONITORS:
        logger.warning(f"[PARALLEL] Monitor {monitor_id} no longer exists, removing from scheduler")
        try:
            scheduler.remove_job(f"parallel_monitor_{monitor_id}")
        except:
            pass
        return
    
    monitor = PARALLEL_MONITORS[monitor_id]
    if not parallel_api or not PARALLEL_MONITOR_ENABLED:
        return
    
    try:
        urls = monitor.get("urls", [])
        objective = monitor.get("objective", "")
        
        if not urls:
            logger.warning(f"[PARALLEL] Monitor {monitor_id} has no URLs")
            return
        
        logger.info(f"[PARALLEL] Checking monitor '{monitor.get('name', monitor_id)}' - {len(urls)} URL(s)")
        result = parallel_api.extract(urls, objective, excerpts=True, full_content=True)
        
        if "results" in result and result["results"]:
            content = result["results"][0].get("full_content", "")
            previous_content = monitor.get("last_content", "")
            
            if content != previous_content or monitor.get("check_count", 0) == 0:
                monitor["last_content"] = content
                monitor["last_check"] = datetime.now().isoformat()
                monitor["check_count"] = monitor.get("check_count", 0) + 1
                save_parallel_monitors()
                
                alert_msg = f"🔄 Parallel Monitor: {monitor.get('name', monitor_id)}\n"
                
                if monitor.get("type") == "kalshi_calendar":
                    games_data = parse_kalshi_calendar_content(content)
                    live_games = games_data.get("live_games", [])
                    upcoming_games = games_data.get("upcoming_games", [])
                    
                    # Format LIVE GAMES with scores and probabilities
                    if live_games:
                        alert_msg += f"\n🔥 LIVE GAMES ({len(live_games)}):\n\n"
                        
                        for game in live_games[:15]:
                            teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                            sport = game.get("sport", "Unknown Sport")
                            quarter = game.get("quarter", "Unknown")
                            
                            alert_msg += f"  {sport}: {teams_str}\n"
                            
                            # SCORE (most important for live games)
                            if game.get("score"):
                                alert_msg += f"    📊 Score: {game['score']}\n"
                            else:
                                alert_msg += f"    📊 Score: Not available\n"
                            
                            # STATUS
                            alert_msg += f"    ⏱️  Status: {quarter}"
                            if game.get("time"):
                                alert_msg += f" - {game['time']}"
                            alert_msg += "\n"
                            
                            # PROBABILITIES - Calendar probabilities
                            calendar_probs = []
                            if game.get("yes_probability"):
                                calendar_probs.append(f"Yes: {game['yes_probability']}%")
                            if game.get("no_probability"):
                                calendar_probs.append(f"No: {game['no_probability']}%")
                            if game.get("team1_probability") and game.get("team2_probability"):
                                calendar_probs.append(f"{game['teams'][0]}: {game['team1_probability']}% | {game['teams'][1]}: {game['team2_probability']}%")
                            
                            if calendar_probs:
                                alert_msg += f"    📈 Calendar Probabilities: {', '.join(calendar_probs)}\n"
                            
                            # PROBABILITIES - Kalshi API market prices
                            if game.get("teams") and kalshi_api:
                                try:
                                    markets_resp = kalshi_api.get_markets(limit=50)
                                    markets = markets_resp.get("markets", [])
                                    markets = filter_active_markets(markets)
                                    
                                    matching_markets = [
                                        m for m in markets
                                        if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                    ]
                                    
                                    if matching_markets:
                                        market = matching_markets[0]
                                        ticker = market.get("ticker")
                                        orderbook = kalshi_api.get_orderbook(ticker)
                                        
                                        yes_bids = orderbook.get("yes_bids", [])
                                        yes_asks = orderbook.get("yes_asks", [])
                                        if yes_bids and yes_asks:
                                            market_prob = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
                                            spread = yes_asks[0].get("price", 100) - yes_bids[0].get("price", 0)
                                            alert_msg += format_kalshi_odds(orderbook, ticker)
                                            
                                            # Alpha detection
                                            alpha = generate_alpha_insight_with_llm(game, market, orderbook, use_search=True)
                                            if alpha:
                                                alert_msg += f"    {alpha}\n"
                                    else:
                                        alert_msg += f"    💰 Kalshi Market: No matching market found\n"
                                except Exception as e:
                                    logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                    alert_msg += f"    💰 Kalshi Market: Error fetching data\n"
                            
                            alert_msg += "\n"
                    
                    # Format UPCOMING GAMES with times and probabilities
                    if upcoming_games:
                        alert_msg += f"\n⏰ UPCOMING GAMES ({len(upcoming_games)}):\n\n"
                        
                        for game in upcoming_games[:15]:
                            teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                            sport = game.get("sport", "Unknown Sport")
                            start_time = game.get("start_time", "Time TBD")
                            
                            alert_msg += f"  {sport}: {teams_str}\n"
                            alert_msg += f"    ⏰ Start Time: {start_time}\n"
                            
                            # PROBABILITIES - Calendar probabilities
                            calendar_probs = []
                            if game.get("yes_probability"):
                                calendar_probs.append(f"Yes: {game['yes_probability']}%")
                            if game.get("no_probability"):
                                calendar_probs.append(f"No: {game['no_probability']}%")
                            if game.get("team1_probability") and game.get("team2_probability"):
                                calendar_probs.append(f"{game['teams'][0]}: {game['team1_probability']}% | {game['teams'][1]}: {game['team2_probability']}%")
                            
                            if calendar_probs:
                                alert_msg += f"    📈 Calendar Probabilities: {', '.join(calendar_probs)}\n"
                            
                            # PROBABILITIES - Kalshi API market prices
                            if game.get("teams") and kalshi_api:
                                try:
                                    markets_resp = kalshi_api.get_markets(limit=50)
                                    markets = markets_resp.get("markets", [])
                                    markets = filter_active_markets(markets)
                                    
                                    matching_markets = [
                                        m for m in markets
                                        if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                    ]
                                    
                                    if matching_markets:
                                        market = matching_markets[0]
                                        ticker = market.get("ticker")
                                        orderbook = kalshi_api.get_orderbook(ticker)
                                        
                                        yes_bids = orderbook.get("yes_bids", [])
                                        yes_asks = orderbook.get("yes_asks", [])
                                        if yes_bids and yes_asks:
                                            market_prob = (yes_bids[0].get("price", 0) + yes_asks[0].get("price", 100)) / 2
                                            spread = yes_asks[0].get("price", 100) - yes_bids[0].get("price", 0)
                                            alert_msg += format_kalshi_odds(orderbook, ticker)
                                    else:
                                        alert_msg += f"    💰 Kalshi Market: No matching market found\n"
                                except Exception as e:
                                    logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                    alert_msg += f"    💰 Kalshi Market: Error fetching data\n"
                            
                            alert_msg += "\n"
                    
                    if not live_games and not upcoming_games:
                        alert_msg += "\n⚠️ No games found in calendar data\n"
                
                PARALLEL_ALERTS.append({
                    "monitor_id": monitor_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": alert_msg,
                    "content_preview": content[:500]
                })
                
                logger.info(f"[PARALLEL ALERT] Monitor '{monitor.get('name', monitor_id)}' triggered alert")
                print(f"[PARALLEL ALERT] {alert_msg}")
    except Exception as e:
        logger.error(f"[PARALLEL ERROR] Failed to check monitor {monitor_id}: {e}")
        print(f"[PARALLEL ERROR] Failed to check monitor {monitor_id}: {e}")


def schedule_parallel_monitor(monitor_id: str):
    """Schedule a single Parallel monitor with its own interval."""
    if monitor_id not in PARALLEL_MONITORS:
        return
    
    monitor = PARALLEL_MONITORS[monitor_id]
    interval = monitor.get("interval", PARALLEL_MONITOR_INTERVAL)
    job_id = f"parallel_monitor_{monitor_id}"
    
    try:
        scheduler.add_job(
            check_single_parallel_monitor,
            'interval',
            seconds=interval,
            args=[monitor_id],
            id=job_id,
            replace_existing=True
        )
        logger.info(f"[PARALLEL] Scheduled monitor '{monitor.get('name', monitor_id)}' with interval {interval}s")
    except Exception as e:
        logger.error(f"[PARALLEL] Failed to schedule monitor {monitor_id}: {e}")


def filter_active_markets(markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter markets to only include active/open markets that haven't resolved yet."""
    today = datetime.now().date()
    active_markets = []
    
    for market in markets:
        status = market.get("status", "").lower()
        
        if status not in ["open", "active"]:
            continue
        
        expiration_time = market.get("expiration_time")
        if expiration_time:
            try:
                if isinstance(expiration_time, str):
                    exp_dt = datetime.fromisoformat(expiration_time.replace("Z", "+00:00"))
                else:
                    exp_dt = expiration_time
                
                if exp_dt.date() < today:
                    continue
            except:
                pass
        
        close_time = market.get("close_time")
        if close_time:
            try:
                if isinstance(close_time, str):
                    close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                else:
                    close_dt = close_time
                
                if close_dt.date() < today:
                    continue
            except:
                pass
        
        active_markets.append(market)
    
    return active_markets


# =========================
# OPENAI (fast, short)
# =========================
def openai_generate(call_sid: str, user_text: str) -> str:
    ctx = CALL_CONTEXT[call_sid]
    topic = ctx["topic"]
    history = ctx["history"][-4:]

    transcript = ""
    for role, text in history:
        transcript += f"{role.upper()}: {text}\n"

    payload = {
        "model": "gpt-4.1-mini",
        "instructions": (
            "Respond immediately.\n"
            "Use 1 short sentence.\n"
            "Be human.\n"
            "Do not mention AI.\n"
            f"Topic: {topic}\n"
        ),
        "input": transcript + f"USER: {user_text}\nASSISTANT:",
        "max_output_tokens": 60,
        "temperature": 0.8,
    }

    logger.info(f"[API CALL] OpenAI generate - call_sid: {call_sid}, topic: '{topic}', user_text length: {len(user_text)}")
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=10,
    )

    resp.raise_for_status()
    data = resp.json()
    text = data["output"][0]["content"][0]["text"].strip()

    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text


# =========================
# ELEVENLABS (blocking but pre-used)
# =========================
def elevenlabs_tts(text: str) -> bytes:
    logger.info(f"[API CALL] ElevenLabs TTS - text length: {len(text)}, voice_id: {ELEVENLABS_VOICE_ID}")
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
            },
        },
        timeout=30,
    )

    resp.raise_for_status()
    return resp.content


# =========================
# MCP TOOLS
# =========================
@mcp.tool(description="Start an AI phone call (ultra low latency)")
def start_agent_call(to: str, topic: str) -> str:
    pending_id = str(int(now() * 1000))

    # PRE-GENERATE GREETING AUDIO
    greeting_text = greeting_for_topic(topic)
    greeting_audio = elevenlabs_tts(greeting_text)
    write_audio(f"pending_{pending_id}", greeting_audio)

    CALL_CONTEXT[f"PENDING:{pending_id}"] = {
        "topic": topic,
        "history": [],
        "start_time": now(),
        "pending_id": pending_id,
    }

    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle?pending_id={pending_id}",
        method="POST",
        barge_in=True,
        timeout=2,
        speech_timeout=0,
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/pending/{pending_id}")
    vr.append(gather)

    logger.info(f"[API CALL] Twilio create call - to: {to}, topic: '{topic}', from: {TWILIO_PHONE_NUMBER}")
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(vr),
        record=True,
        recording_channels="dual",
    )
    logger.info(f"[API CALL] Twilio call created - call_sid: {call.sid}, status: {call.status}")

    CALL_CONTEXT[call.sid] = {
        "topic": topic,
        "history": [],
        "start_time": now(),
        "pending_id": pending_id,
    }

    return f"Call started ({call.sid})"


@mcp.tool(description="Search Kalshi markets by keyword")
def search_kalshi_markets(query: str, limit: int = 20) -> str:
    """Search for Kalshi markets matching a query."""
    logger.info(f"[MCP TOOL] search_kalshi_markets called - query: '{query}', limit: {limit}")
    if not kalshi_api:
        return "Error: Kalshi API not configured. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY environment variables."
    
    try:
        markets_response = kalshi_api.get_markets(limit=limit)
        markets = markets_response.get("markets", [])
        markets = filter_active_markets(markets)
        logger.info(f"[KALSHI] search_kalshi_markets - found {len(markets)} active markets after filtering")
        
        query_lower = query.lower()
        matches = []
        for market in markets:
            title = (market.get("title") or "").lower()
            ticker = (market.get("ticker") or "").lower()
            subtitle = (market.get("subtitle") or "").lower()
            
            if query_lower in title or query_lower in ticker or query_lower in subtitle:
                matches.append(market)
        
        if not matches:
            return f"No markets found matching '{query}'"
        
        result = f"Found {len(matches)} markets matching '{query}':\n\n"
        for i, market in enumerate(matches[:limit], 1):
            result += f"{i}. {market.get('ticker')}: {market.get('title')}\n"
            result += f"   Status: {market.get('status')}\n"
            if market.get('subtitle'):
                result += f"   {market.get('subtitle')}\n"
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error searching markets: {str(e)}"


@mcp.tool(description="Get detailed market data, orderbook, and price movement analysis for a Kalshi market ticker")
def get_kalshi_market(ticker: str) -> str:
    """Get market data, orderbook, and price movement trends for a specific ticker."""
    logger.info(f"[MCP TOOL] get_kalshi_market called - ticker: '{ticker}'")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    try:
        market_data = kalshi_api.get_market(ticker)
        orderbook = kalshi_api.get_orderbook(ticker)
        
        yes_bids = orderbook.get("yes_bids", [])
        yes_asks = orderbook.get("yes_asks", [])
        no_bids = orderbook.get("no_bids", [])
        no_asks = orderbook.get("no_asks", [])
        
        mid_price = None
        bid_price = None
        ask_price = None
        
        if yes_bids and yes_asks:
            bid_price = yes_bids[0].get("price", 0)
            ask_price = yes_asks[0].get("price", 100)
            mid_price = (bid_price + ask_price) / 2
        elif yes_bids:
            bid_price = yes_bids[0].get("price", 0)
            mid_price = bid_price
        elif yes_asks:
            ask_price = yes_asks[0].get("price", 100)
            mid_price = ask_price
        
        spread = (ask_price - bid_price) if (bid_price and ask_price) else None
        
        total_bid_size = sum(b.get("size", 0) for b in yes_bids[:5])
        total_ask_size = sum(a.get("size", 0) for a in yes_asks[:5])
        bid_ask_ratio = total_bid_size / total_ask_size if total_ask_size > 0 else None
        
        now_ts = datetime.now().isoformat()
        
        if ticker not in KALSHI_PRICE_HISTORY:
            KALSHI_PRICE_HISTORY[ticker] = []
        
        KALSHI_PRICE_HISTORY[ticker].append({
            "timestamp": now_ts,
            "mid_price": mid_price,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "spread": spread
        })
        
        if len(KALSHI_PRICE_HISTORY[ticker]) > 20:
            KALSHI_PRICE_HISTORY[ticker] = KALSHI_PRICE_HISTORY[ticker][-20:]
        
        price_history = KALSHI_PRICE_HISTORY[ticker]
        price_movement = ""
        
        if len(price_history) >= 2:
            prev_price = price_history[-2].get("mid_price")
            if prev_price and mid_price:
                change = mid_price - prev_price
                change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                price_movement = f"\nPrice Movement:\n  {direction} {abs(change):.2f} cents ({abs(change_pct):.2f}%)"
                price_movement += f" from {prev_price:.2f} cents"
        
        if len(price_history) >= 3:
            recent_prices = [p.get("mid_price") for p in price_history[-3:] if p.get("mid_price")]
            if len(recent_prices) == 3:
                trend = "upward" if recent_prices[2] > recent_prices[0] else "downward" if recent_prices[2] < recent_prices[0] else "sideways"
                price_movement += f"\n  Trend: {trend} over last 3 checks"
        
        momentum_indicator = ""
        if bid_ask_ratio:
            if bid_ask_ratio > 1.5:
                momentum_indicator = "🟢 Strong buying pressure (more bids than asks)"
            elif bid_ask_ratio < 0.67:
                momentum_indicator = "🔴 Strong selling pressure (more asks than bids)"
            else:
                momentum_indicator = "🟡 Balanced (similar bid/ask depth)"
        
        result = f"Market: {market_data.get('title', 'N/A')}\n"
        result += f"Ticker: {ticker}\n"
        result += f"Status: {market_data.get('status', 'N/A')}\n"
        if market_data.get('subtitle'):
            result += f"Subtitle: {market_data.get('subtitle')}\n"
        
        result += f"\n📊 Current Pricing:\n"
        result += f"  Mid Price: {mid_price:.2f} cents\n"
        if bid_price and ask_price:
            result += f"  Bid: {bid_price} | Ask: {ask_price}\n"
            result += f"  Spread: {spread:.2f} cents\n"
        
        result += f"\n📈 Orderbook Depth:\n"
        result += f"  Yes Bids: {len(yes_bids)} orders | Yes Asks: {len(yes_asks)} orders\n"
        result += f"  No Bids: {len(no_bids)} orders | No Asks: {len(no_asks)} orders\n"
        
        if yes_bids:
            result += f"  Top Yes Bid: {yes_bids[0].get('price')} @ {yes_bids[0].get('size')} contracts\n"
        if yes_asks:
            result += f"  Top Yes Ask: {yes_asks[0].get('price')} @ {yes_asks[0].get('size')} contracts\n"
        
        if total_bid_size > 0 or total_ask_size > 0:
            result += f"  Top 5 Depth: {total_bid_size} bids vs {total_ask_size} asks\n"
        
        if momentum_indicator:
            result += f"\n{momentum_indicator}\n"
        
        if price_movement:
            result += price_movement
        
        if len(price_history) > 1:
            prices = [p.get("mid_price") for p in price_history if p.get("mid_price")]
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                result += f"\n\n📉 Price Range (this session):\n"
                result += f"  Low: {min_price:.2f} cents | High: {max_price:.2f} cents\n"
                result += f"  Range: {max_price - min_price:.2f} cents"
        
        volume_data = analyze_orderbook_volume_distribution(orderbook)
        if ticker not in KALSHI_VOLUME_HISTORY:
            KALSHI_VOLUME_HISTORY[ticker] = []
        
        KALSHI_VOLUME_HISTORY[ticker].append({
            "timestamp": datetime.now().isoformat(),
            "total_volume": volume_data["total_volume"],
            "yes_bid_volume": volume_data["yes_bid_volume"],
            "yes_ask_volume": volume_data["yes_ask_volume"]
        })
        
        if len(KALSHI_VOLUME_HISTORY[ticker]) > 20:
            KALSHI_VOLUME_HISTORY[ticker] = KALSHI_VOLUME_HISTORY[ticker][-20:]
        
        return result
    except Exception as e:
        return f"Error getting market: {str(e)}"


@mcp.tool(description="Get AI-generated betting insights for a Kalshi market")
def get_kalshi_insights(ticker: str) -> str:
    """Get AI-generated betting insights for a market."""
    logger.info(f"[MCP TOOL] get_kalshi_insights called - ticker: '{ticker}'")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    try:
        market_data = kalshi_api.get_market(ticker)
        orderbook = kalshi_api.get_orderbook(ticker)
        
        insights = generate_market_insights(ticker, market_data, orderbook)
        if not insights:
            return f"Could not generate insights for {ticker}. Market data retrieved successfully."
        
        return f"AI Insights for {ticker} ({market_data.get('title', 'N/A')}):\n\n{insights}"
    except Exception as e:
        return f"Error getting insights: {str(e)}"


@mcp.tool(description="Watch a Kalshi market for price alerts")
def watch_kalshi_market(ticker: str, alert_price: int, direction: str = "above", remove_after_trigger: bool = False) -> str:
    """Set up a price alert for a market. Direction: 'above' or 'below'."""
    logger.info(f"[MCP TOOL] watch_kalshi_market called - ticker: '{ticker}', alert_price: {alert_price}, direction: '{direction}', remove_after_trigger: {remove_after_trigger}")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    if direction not in ["above", "below"]:
        return "Error: direction must be 'above' or 'below'"
    
    try:
        market_data = kalshi_api.get_market(ticker)
        
        KALSHI_WATCHES[ticker] = {
            "ticker": ticker,
            "title": market_data.get("title", ticker),
            "alert_price": alert_price,
            "direction": direction,
            "remove_after_trigger": remove_after_trigger,
            "created_at": datetime.now().isoformat()
        }
        save_kalshi_watches()
        
        return f"Watching {ticker} ({market_data.get('title', 'N/A')}) - Alert when price goes {direction} {alert_price} cents"
    except Exception as e:
        return f"Error setting watch: {str(e)}"


@mcp.tool(description="List all watched Kalshi markets")
def list_kalshi_watches() -> str:
    """List all markets being watched for alerts."""
    logger.info(f"[MCP TOOL] list_kalshi_watches called")
    if not KALSHI_WATCHES:
        return "No markets being watched."
    
    result = f"Watching {len(KALSHI_WATCHES)} markets:\n\n"
    for ticker, watch in KALSHI_WATCHES.items():
        result += f"• {ticker}: {watch.get('title', 'N/A')}\n"
        result += f"  Alert: {watch.get('direction')} {watch.get('alert_price')} cents\n"
        result += f"  Created: {watch.get('created_at', 'N/A')}\n\n"
    
    return result


@mcp.tool(description="Remove a market from watch list")
def unwatch_kalshi_market(ticker: str) -> str:
    """Stop watching a market for alerts."""
    logger.info(f"[MCP TOOL] unwatch_kalshi_market called - ticker: '{ticker}'")
    if ticker not in KALSHI_WATCHES:
        return f"{ticker} is not being watched."
    
    del KALSHI_WATCHES[ticker]
    save_kalshi_watches()
    return f"Stopped watching {ticker}"


@mcp.tool(description="Get recent Kalshi market insights and alerts")
def get_kalshi_recent_insights(limit: int = 10) -> str:
    """Get recent AI insights and price alerts."""
    logger.info(f"[MCP TOOL] get_kalshi_recent_insights called - limit: {limit}")
    if not KALSHI_INSIGHTS:
        return "No recent insights or alerts."
    
    recent = KALSHI_INSIGHTS[-limit:]
    result = f"Recent insights ({len(recent)}):\n\n"
    
    for insight in recent:
        result += f"[{insight.get('timestamp', 'N/A')}] {insight.get('message', 'N/A')}\n\n"
    
    return result


@mcp.tool(description="Find interesting Kalshi markets right now and generate AI insights")
def find_interesting_kalshi_markets(limit: int = 5, search_query: str = "") -> str:
    """Find interesting markets based on liquidity and activity, then generate AI insights for them."""
    logger.info(f"[MCP TOOL] find_interesting_kalshi_markets called - limit: {limit}, search_query: '{search_query}'")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    try:
        markets_response = kalshi_api.get_markets(limit=100)
        markets = markets_response.get("markets", [])
        markets = filter_active_markets(markets)
        logger.info(f"[KALSHI] find_interesting_kalshi_markets - found {len(markets)} active markets after filtering")
        
        if search_query:
            query_lower = search_query.lower()
            markets = [
                m for m in markets
                if query_lower in (m.get("title") or "").lower() or
                   query_lower in (m.get("ticker") or "").lower() or
                   query_lower in (m.get("subtitle") or "").lower()
            ]
        
        interesting_markets = []
        
        for market in markets:
            ticker = market.get("ticker")
            
            try:
                orderbook = kalshi_api.get_orderbook(ticker)
                yes_bids = orderbook.get("yes_bids", [])
                yes_asks = orderbook.get("yes_asks", [])
                
                if not yes_bids or not yes_asks:
                    continue
                
                bid_price = yes_bids[0].get("price", 0)
                ask_price = yes_asks[0].get("price", 100)
                mid_price = (bid_price + ask_price) / 2
                spread = ask_price - bid_price
                
                if mid_price < 5 or mid_price > 95:
                    continue
                
                if spread > 20:
                    continue
                
                interesting_markets.append({
                    "market": market,
                    "ticker": ticker,
                    "mid_price": mid_price,
                    "spread": spread,
                    "orderbook": orderbook
                })
            except:
                continue
            
            if len(interesting_markets) >= limit * 2:
                break
        
        if not interesting_markets:
            return "No interesting markets found matching the criteria. Try a different search query or check back later."
        
        interesting_markets.sort(key=lambda x: x["spread"])
        top_markets = interesting_markets[:limit]
        
        result = f"Found {len(top_markets)} interesting markets:\n\n"
        
        for i, market_info in enumerate(top_markets, 1):
            market = market_info["market"]
            ticker = market_info["ticker"]
            mid_price = market_info["mid_price"]
            spread = market_info["spread"]
            orderbook = market_info["orderbook"]
            
            result += f"{i}. {ticker}: {market.get('title', 'N/A')}\n"
            result += f"   Price: {mid_price:.1f} cents | Spread: {spread:.1f} cents\n"
            
            if KALSHI_INSIGHTS_ENABLED:
                insights = generate_market_insights(ticker, market, orderbook)
                if insights:
                    result += f"   💡 Insight: {insights}\n"
            
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error finding interesting markets: {str(e)}"


@mcp.tool(description="Compare multiple Kalshi markets side-by-side with value analysis and AI recommendations")
def compare_kalshi_markets(tickers: str) -> str:
    """Compare 2-5 markets. Provide tickers as comma-separated string."""
    logger.info(f"[MCP TOOL] compare_kalshi_markets called - tickers: '{tickers}'")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    ticker_list = [t.strip() for t in tickers.split(",")]
    if len(ticker_list) < 2 or len(ticker_list) > 5:
        return "Error: Provide 2-5 tickers separated by commas."
    
    try:
        markets_data = []
        for ticker in ticker_list:
            try:
                market_data = kalshi_api.get_market(ticker)
                orderbook = kalshi_api.get_orderbook(ticker)
                volume_data = analyze_orderbook_volume_distribution(orderbook)
                risk_scores = calculate_risk_scores(ticker, market_data, orderbook, volume_data)
                
                yes_bids = orderbook.get("yes_bids", [])
                yes_asks = orderbook.get("yes_asks", [])
                mid_price = None
                spread = None
                if yes_bids and yes_asks:
                    bid_price = yes_bids[0].get("price", 0)
                    ask_price = yes_asks[0].get("price", 100)
                    mid_price = (bid_price + ask_price) / 2
                    spread = ask_price - bid_price
                
                markets_data.append({
                    "ticker": ticker,
                    "market_data": market_data,
                    "mid_price": mid_price,
                    "spread": spread,
                    "volume_data": volume_data,
                    "risk_scores": risk_scores
                })
            except Exception as e:
                markets_data.append({"ticker": ticker, "error": str(e)})
        
        result = "📊 Market Comparison:\n\n"
        
        for i, m in enumerate(markets_data, 1):
            if "error" in m:
                result += f"{i}. {m['ticker']}: Error - {m['error']}\n\n"
                continue
            
            result += f"{i}. {m['ticker']}: {m['market_data'].get('title', 'N/A')}\n"
            result += f"   Price: {m['mid_price']:.2f} cents | Spread: {m['spread']:.2f} cents\n"
            result += f"   Volume: {m['volume_data']['total_volume']} contracts\n"
            result += f"   Risk Score: {m['risk_scores'].get('overall_risk', 'N/A')}/100\n"
            result += f"   Liquidity: {m['risk_scores'].get('liquidity_score', 'N/A')}/100\n"
            
            if m['volume_data']['max_bid_concentration'] > 50:
                result += f"   ⚠️ High bid concentration: {m['volume_data']['max_bid_concentration']:.1f}%\n"
            if m['volume_data']['max_ask_concentration'] > 50:
                result += f"   ⚠️ High ask concentration: {m['volume_data']['max_ask_concentration']:.1f}%\n"
            
            result += "\n"
        
        best_spread = min([m for m in markets_data if "error" not in m], key=lambda x: x.get("spread", 999), default=None)
        best_liquidity = max([m for m in markets_data if "error" not in m], key=lambda x: x.get("risk_scores", {}).get("liquidity_score", 0), default=None)
        lowest_risk = min([m for m in markets_data if "error" not in m], key=lambda x: x.get("risk_scores", {}).get("overall_risk", 999), default=None)
        
        if best_spread and best_liquidity and lowest_risk:
            result += "\n💡 Recommendations:\n"
            result += f"  • Tightest Spread: {best_spread['ticker']} ({best_spread['spread']:.2f} cents) - Best for quick entry/exit\n"
            result += f"  • Best Liquidity: {best_liquidity['ticker']} (Score: {best_liquidity['risk_scores']['liquidity_score']:.1f}) - Easiest to trade\n"
            result += f"  • Lowest Risk: {lowest_risk['ticker']} (Risk: {lowest_risk['risk_scores']['overall_risk']:.1f}) - Most stable\n"
        
        return result
    except Exception as e:
        return f"Error comparing markets: {str(e)}"


@mcp.tool(description="Get comprehensive risk analysis including volume distribution, concentration risk, and volume shifts")
def get_kalshi_risk_analysis(ticker: str) -> str:
    """Deep risk analysis with volume distribution, concentration risk, and historical trends."""
    logger.info(f"[MCP TOOL] get_kalshi_risk_analysis called - ticker: '{ticker}'")
    if not kalshi_api:
        return "Error: Kalshi API not configured."
    
    try:
        market_data = kalshi_api.get_market(ticker)
        orderbook = kalshi_api.get_orderbook(ticker)
        volume_data = analyze_orderbook_volume_distribution(orderbook)
        risk_scores = calculate_risk_scores(ticker, market_data, orderbook, volume_data)
        volume_shifts = analyze_volume_shifts(ticker)
        
        result = f"🔍 Risk Analysis: {ticker} ({market_data.get('title', 'N/A')})\n\n"
        
        result += "📊 Risk Scores:\n"
        result += f"  Overall Risk: {risk_scores.get('overall_risk', 'N/A')}/100\n"
        result += f"  Volatility: {risk_scores.get('volatility_score', 'N/A')}/100\n"
        result += f"  Liquidity: {risk_scores.get('liquidity_score', 'N/A')}/100\n"
        result += f"  Concentration Risk: {risk_scores.get('concentration_risk', 'N/A')}/100\n"
        result += f"  Volume Stability: {risk_scores.get('volume_stability', 'N/A')}/100\n\n"
        
        result += "📈 Volume Distribution:\n"
        result += f"  Total Volume: {volume_data['total_volume']} contracts\n"
        result += f"  Yes Bids: {volume_data['yes_bid_volume']} | Yes Asks: {volume_data['yes_ask_volume']}\n"
        result += f"  No Bids: {volume_data['no_bid_volume']} | No Asks: {volume_data['no_ask_volume']}\n"
        
        if volume_data['buyer_seller_ratio']:
            ratio = volume_data['buyer_seller_ratio']
            result += f"  Buyer/Seller Ratio: {ratio:.2f}x\n"
            if ratio > 1.5:
                result += "    → More buying pressure\n"
            elif ratio < 0.67:
                result += "    → More selling pressure\n"
        
        result += "\n⚠️ Concentration Risk:\n"
        result += f"  Largest Bid Order: {volume_data['max_bid_concentration']:.1f}% of bid volume\n"
        result += f"  Largest Ask Order: {volume_data['max_ask_concentration']:.1f}% of ask volume\n"
        
        if volume_data['max_bid_concentration'] > 70:
            result += f"  🚨 WARNING: {volume_data['max_bid_concentration']:.1f}% of bid volume in single order - high manipulation risk\n"
        if volume_data['max_ask_concentration'] > 70:
            result += f"  🚨 WARNING: {volume_data['max_ask_concentration']:.1f}% of ask volume in single order - high manipulation risk\n"
        
        if volume_data['max_bid_concentration'] > 50 or volume_data['max_ask_concentration'] > 50:
            result += "  → Market may be dominated by one large player\n"
        
        result += "\n📉 Top 5 Bid Distribution:\n"
        for i, bid in enumerate(volume_data['yes_bid_distribution'][:5], 1):
            result += f"  {i}. {bid['size']} contracts @ {bid['price']} cents ({bid['pct']:.1f}% of bid volume)\n"
        
        result += "\n📈 Top 5 Ask Distribution:\n"
        for i, ask in enumerate(volume_data['yes_ask_distribution'][:5], 1):
            result += f"  {i}. {ask['size']} contracts @ {ask['price']} cents ({ask['pct']:.1f}% of ask volume)\n"
        
        if "error" not in volume_shifts:
            result += "\n📊 Volume Trends:\n"
            result += f"  Recent Avg: {volume_shifts['recent_avg_volume']:.0f} contracts\n"
            result += f"  Older Avg: {volume_shifts['older_avg_volume']:.0f} contracts\n"
            result += f"  Change: {volume_shifts['volume_change_pct']:+.1f}%\n"
            result += f"  Trend: {volume_shifts['trend']}\n"
            result += f"  Current: {volume_shifts['current_volume']} contracts\n"
        
        return result
    except Exception as e:
        return f"Error getting risk analysis: {str(e)}"


@mcp.tool(description="Create a general Parallel API monitor for any URL(s). Monitor interval is in seconds (e.g., 300 for 5 minutes).")
def create_parallel_monitor(name: str, urls: str, objective: str, monitor_interval: int, monitor_type: str = "general") -> str:
    """Create a monitor for Parallel API extraction. URLs should be comma-separated. Monitor interval is in seconds."""
    logger.info(f"[MCP TOOL] create_parallel_monitor called - name: '{name}', type: '{monitor_type}', interval: {monitor_interval}s, urls: {len(urls.split(','))}")
    if not parallel_api:
        return "Error: Parallel API not configured. Set PARALLEL_API_KEY environment variable."
    
    if monitor_interval < 60:
        return "Error: Monitor interval must be at least 60 seconds (1 minute)."
    
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    if not url_list:
        return "Error: Provide at least one URL."
    
    monitor_id = f"monitor_{int(time.time())}"
    PARALLEL_MONITORS[monitor_id] = {
        "name": name,
        "urls": url_list,
        "objective": objective,
        "type": monitor_type,
        "interval": monitor_interval,
        "created_at": datetime.now().isoformat(),
        "check_count": 0,
        "last_content": "",
        "last_check": None
    }
    save_parallel_monitors()
    
    # Schedule this monitor
    schedule_parallel_monitor(monitor_id)
    
    return f"Created monitor '{name}' (ID: {monitor_id}) for {len(url_list)} URL(s). Will check every {monitor_interval}s."


@mcp.tool(description="Set up Kalshi calendar monitoring to get live game alerts with alpha insights. Monitor interval is in seconds (e.g., 300 for 5 minutes).")
def setup_kalshi_calendar_monitor(monitor_interval: int) -> str:
    """Set up automatic monitoring of Kalshi calendar page for live games. Monitor interval is in seconds."""
    logger.info(f"[MCP TOOL] setup_kalshi_calendar_monitor called - interval: {monitor_interval}s")
    if not parallel_api:
        return "Error: Parallel API not configured."
    
    if monitor_interval < 60:
        return "Error: Monitor interval must be at least 60 seconds (1 minute)."
    
    monitor_id = "kalshi_calendar"
    PARALLEL_MONITORS[monitor_id] = {
        "name": "Kalshi Calendar - Live Games",
        "urls": ["https://kalshi.com/calendar"],
        "objective": "Extract all live game information from Kalshi calendar page including market names, prices, categories, live market data, game status, teams, and probabilities",
        "type": "kalshi_calendar",
        "interval": monitor_interval,
        "created_at": datetime.now().isoformat(),
        "check_count": 0,
        "last_content": "",
        "last_check": None
    }
    save_parallel_monitors()
    
    # Schedule this monitor
    schedule_parallel_monitor(monitor_id)
    
    return f"✅ Kalshi calendar monitoring enabled! Will check every {monitor_interval}s for live games and compare with market probabilities to find alpha opportunities."


@mcp.tool(description="List all active Parallel monitors")
def list_parallel_monitors() -> str:
    """List all active Parallel API monitors."""
    logger.info(f"[MCP TOOL] list_parallel_monitors called")
    if not PARALLEL_MONITORS:
        return "No active monitors."
    
    result = f"Active Monitors ({len(PARALLEL_MONITORS)}):\n\n"
    for monitor_id, monitor in PARALLEL_MONITORS.items():
        result += f"• {monitor.get('name', monitor_id)}\n"
        result += f"  Type: {monitor.get('type', 'general')}\n"
        result += f"  URLs: {len(monitor.get('urls', []))}\n"
        result += f"  Checks: {monitor.get('check_count', 0)}\n"
        result += f"  Interval: {monitor.get('interval', PARALLEL_MONITOR_INTERVAL)}s\n"
        if monitor.get('last_check'):
            result += f"  Last Check: {monitor['last_check']}\n"
        result += "\n"
    
    return result


@mcp.tool(description="Remove a Parallel monitor")
def remove_parallel_monitor(monitor_id: str) -> str:
    """Remove a Parallel API monitor."""
    logger.info(f"[MCP TOOL] remove_parallel_monitor called - monitor_id: '{monitor_id}'")
    if monitor_id not in PARALLEL_MONITORS:
        return f"Monitor '{monitor_id}' not found."
    
    # Unschedule the job
    job_id = f"parallel_monitor_{monitor_id}"
    try:
        scheduler.remove_job(job_id)
        logger.info(f"[PARALLEL] Removed scheduled job for monitor {monitor_id}")
    except:
        pass
    
    del PARALLEL_MONITORS[monitor_id]
    save_parallel_monitors()
    return f"Removed monitor '{monitor_id}'"


@mcp.tool(description="Get recent Parallel monitor alerts and insights")
def get_parallel_alerts(limit: int = 10) -> str:
    """Get recent alerts from Parallel monitors."""
    logger.info(f"[MCP TOOL] get_parallel_alerts called - limit: {limit}")
    if not PARALLEL_ALERTS:
        return "No recent alerts."
    
    recent = PARALLEL_ALERTS[-limit:]
    result = f"Recent Parallel Alerts ({len(recent)}):\n\n"
    
    for alert in recent:
        result += f"[{alert.get('timestamp', 'N/A')}]\n"
        result += f"{alert.get('message', 'N/A')}\n"
        result += "\n" + "="*50 + "\n\n"
    
    return result


@mcp.tool(description="Manually trigger a Parallel monitor check")
def check_parallel_monitor(monitor_id: str) -> str:
    """Manually check a specific Parallel monitor."""
    logger.info(f"[MCP TOOL] check_parallel_monitor called - monitor_id: '{monitor_id}'")
    if monitor_id not in PARALLEL_MONITORS:
        return f"Monitor '{monitor_id}' not found."
    
    monitor = PARALLEL_MONITORS[monitor_id]
    urls = monitor.get("urls", [])
    objective = monitor.get("objective", "")
    
    if not parallel_api:
        return "Error: Parallel API not configured."
    
    try:
        result = parallel_api.extract(urls, objective, excerpts=True, full_content=True)
        
        if "results" in result and result["results"]:
            content = result["results"][0].get("full_content", "")
            
            if monitor.get("type") == "kalshi_calendar":
                games_data = parse_kalshi_calendar_content(content)
                live_games = games_data.get("live_games", [])
                upcoming_games = games_data.get("upcoming_games", [])
                
                response = f"✅ Checked {monitor.get('name')}\n\n"
                
                if live_games:
                    response += f"🔥 LIVE GAMES ({len(live_games)}):\n\n"
                    for game in live_games[:10]:
                        teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                        response += f"• {game.get('sport', 'Unknown')}: {teams_str}\n"
                        if game.get("score"):
                            response += f"  📊 Score: {game['score']}\n"
                        if game.get("quarter"):
                            response += f"  ⏱️  Status: {game['quarter']}\n"
                        if game.get("yes_probability") or game.get("no_probability"):
                            probs = []
                            if game.get("yes_probability"):
                                probs.append(f"Yes: {game['yes_probability']}%")
                            if game.get("no_probability"):
                                probs.append(f"No: {game['no_probability']}%")
                            response += f"  📈 Calendar Probabilities: {', '.join(probs)}\n"
                        
                        # Get Kalshi API live odds
                        if game.get("teams") and kalshi_api:
                            try:
                                markets_resp = kalshi_api.get_markets(limit=100)
                                markets = markets_resp.get("markets", [])
                                markets = filter_active_markets(markets)
                                
                                matching_markets = [
                                    m for m in markets
                                    if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                ]
                                
                                if matching_markets:
                                    market = matching_markets[0]
                                    ticker = market.get("ticker")
                                    orderbook = kalshi_api.get_orderbook(ticker)
                                    response += format_kalshi_odds(orderbook, ticker)
                                    
                                    if len(matching_markets) > 1:
                                        response += f"      📊 {len(matching_markets)} total markets available\n"
                                else:
                                    response += f"  💰 Kalshi Market: No matching market found\n"
                            except Exception as e:
                                logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                response += f"  💰 Kalshi Market: Error fetching data\n"
                        
                        response += "\n"
                
                if upcoming_games:
                    response += f"\n⏰ UPCOMING GAMES ({len(upcoming_games)}):\n\n"
                    for game in upcoming_games[:10]:
                        teams_str = " vs ".join(game.get("teams", ["Unknown"]))
                        response += f"• {game.get('sport', 'Unknown')}: {teams_str}\n"
                        if game.get("start_time"):
                            response += f"  ⏰ Start: {game['start_time']}\n"
                        if game.get("yes_probability") or game.get("no_probability"):
                            probs = []
                            if game.get("yes_probability"):
                                probs.append(f"Yes: {game['yes_probability']}%")
                            if game.get("no_probability"):
                                probs.append(f"No: {game['no_probability']}%")
                            response += f"  📈 Calendar Probabilities: {', '.join(probs)}\n"
                        
                        # Get Kalshi API odds for upcoming games
                        if game.get("teams") and kalshi_api:
                            try:
                                markets_resp = kalshi_api.get_markets(limit=100)
                                markets = markets_resp.get("markets", [])
                                markets = filter_active_markets(markets)
                                
                                matching_markets = [
                                    m for m in markets
                                    if any(team.lower() in (m.get("title") or "").lower() for team in game.get("teams", []))
                                ]
                                
                                if matching_markets:
                                    market = matching_markets[0]
                                    ticker = market.get("ticker")
                                    orderbook = kalshi_api.get_orderbook(ticker)
                                    response += format_kalshi_odds(orderbook, ticker)
                                    
                                    if len(matching_markets) > 1:
                                        response += f"      📊 {len(matching_markets)} total markets available\n"
                                else:
                                    response += f"  💰 Kalshi Market: No matching market found\n"
                            except Exception as e:
                                logger.warning(f"[PARALLEL] Failed to get market data: {e}")
                                response += f"  💰 Kalshi Market: Error fetching data\n"
                        
                        response += "\n"
                
                if not live_games and not upcoming_games:
                    response += "⚠️ No games found\n"
                
                return response
            else:
                return f"✅ Checked {monitor.get('name')}\n\nContent length: {len(content)} characters\nPreview: {content[:200]}..."
        
        return f"✅ Checked {monitor.get('name')} - No results returned"
    except Exception as e:
        return f"Error checking monitor: {str(e)}"


# =========================
# ROUTES
# =========================
@mcp.custom_route("/twilio/audio/pending/{pending_id}", methods=["GET"])
async def twilio_audio_pending(request: Request):
    pending_id = request.path_params["pending_id"]
    p = audio_path(f"pending_{pending_id}")
    if not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return Response(p.read_bytes(), media_type="audio/mpeg")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    pending_id = request.query_params.get("pending_id")
    form = await request.form()

    call_sid = form.get("CallSid")
    user_text = (form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if not call_sid:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    if call_sid not in CALL_CONTEXT and pending_id:
        pctx = CALL_CONTEXT.get(f"PENDING:{pending_id}")
        if pctx:
            CALL_CONTEXT[call_sid] = pctx

    ctx = CALL_CONTEXT.get(call_sid)
    if not ctx:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    if now() - ctx["start_time"] > MAX_CALL_DURATION_SECONDS:
        goodbye_audio = elevenlabs_tts("I have to run. Talk soon.")
        key = f"{call_sid}_end"
        write_audio(key, goodbye_audio)
        vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{key}")
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    if not user_text:
        gather = Gather(
            input="speech",
            action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
            method="POST",
            timeout=2,
            speech_timeout=0,
        )
        vr.append(gather)
        return PlainTextResponse(str(vr), media_type="text/xml")

    ctx["history"].append(("user", user_text))

    reply = openai_generate(call_sid, user_text)
    ctx["history"].append(("assistant", reply))

    audio = elevenlabs_tts(reply)
    key = f"{call_sid}_{int(now() * 1000)}"
    write_audio(key, audio)

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        timeout=2,
        speech_timeout=0,
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/{key}")
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{key}", methods=["GET"])
async def twilio_audio(request: Request):
    key = request.path_params["key"]
    p = audio_path(key)
    if not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return Response(p.read_bytes(), media_type="audio/mpeg")


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    if kalshi_api and KALSHI_ALERTS_ENABLED:
        use_cron = os.environ.get("KALSHI_USE_CRON_SCHEDULE", "true").lower() == "true"
        
        if use_cron:
            scheduler.add_job(
                check_kalshi_watches,
                'cron',
                day_of_week=KALSHI_SCHEDULE_DAYS,
                hour=KALSHI_SCHEDULE_HOURS,
                minute=0,
                id='kalshi_watch_check',
                replace_existing=True
            )
            print(f"[KALSHI] Started background monitoring (cron: {KALSHI_SCHEDULE_DAYS} at {KALSHI_SCHEDULE_HOURS}:00)")
        else:
            scheduler.add_job(
                check_kalshi_watches,
                'interval',
                seconds=KALSHI_CHECK_INTERVAL,
                id='kalshi_watch_check',
                replace_existing=True
            )
            print(f"[KALSHI] Started background monitoring (interval: {KALSHI_CHECK_INTERVAL}s)")
    
    if parallel_api and PARALLEL_MONITOR_ENABLED and PARALLEL_MONITORS:
        # Schedule each monitor individually with its own interval
        for monitor_id in PARALLEL_MONITORS.keys():
            schedule_parallel_monitor(monitor_id)
        print(f"[PARALLEL] Started background monitoring for {len(PARALLEL_MONITORS)} monitor(s) with individual intervals")
    
    if scheduler.get_jobs():
        scheduler.start()
    
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
    )
