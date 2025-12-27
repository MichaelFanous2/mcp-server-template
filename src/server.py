#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather


# =========================
# ENVIRONMENT VARIABLES
# =========================

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]


# =========================
# CLIENTS
# =========================

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# =========================
# STORAGE
# =========================

AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CALL_CONTEXT: Dict[str, Dict[str, Any]] = {}

MAX_CALL_DURATION_SECONDS = 240


# =========================
# UTILITIES
# =========================

def now() -> float:
    return time.time()


def log(msg: str):
    print(f"\n=== {msg} ===\n", flush=True)


def write_audio(call_sid: str, turn_id: str, audio: bytes) -> Path:
    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    with open(path, "wb") as f:
        f.write(audio)
    return path


# =========================
# OPENAI
# =========================

def openai_generate(call_sid: str, user_text: str) -> str:
    ctx = CALL_CONTEXT[call_sid]
    topic = ctx["topic"]
    history = ctx["history"]

    transcript = ""
    for role, text in history:
        transcript += f"{role.upper()}: {text}\n"

    instructions = (
        "You are having a live phone conversation.\n"
        "Be natural, concise, confident, and human.\n"
        "Do NOT mention AI, tools, or systems.\n"
        f"Conversation topic: {topic}\n"
    )

    payload = {
        "model": "gpt-4.1-mini",
        "instructions": instructions,
        "input": transcript + f"USER: {user_text}\nASSISTANT:",
        "max_output_tokens": 200,
    }

    log(f"[CALL {call_sid}] OPENAI REQUEST PAYLOAD")
    print(payload, flush=True)

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    log(f"[CALL {call_sid}] OPENAI RAW RESPONSE")
    print(resp.text, flush=True)

    resp.raise_for_status()
    data = resp.json()

    text = data["output"][0]["content"][0]["text"]
    return text.strip()


# =========================
# ELEVENLABS
# =========================

def elevenlabs_tts(text: str) -> bytes:
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.8,
        },
    }

    log("ELEVENLABS REQUEST PAYLOAD")
    print(payload, flush=True)

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    log("ELEVENLABS RAW RESPONSE HEADERS")
    print(resp.headers, flush=True)

    resp.raise_for_status()
    audio = resp.content

    log(f"ELEVENLABS AUDIO BYTES = {len(audio)}")
    return audio


# =========================
# MCP TOOL (POKE CALLS THIS)
# =========================

@mcp.tool(description="Start an AI phone call")
def start_agent_call(to: str, topic: str) -> str:
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=f"{PUBLIC_BASE_URL}/twilio/voice",
        method="GET",
    )

    CALL_CONTEXT[call.sid] = {
        "topic": topic,
        "history": [],
        "start_time": now(),
    }

    log(f"[CALL {call.sid}] CALL STARTED WITH TOPIC")
    print(topic, flush=True)

    return f"Call started ({call.sid})"


# =========================
# TWILIO ROUTES
# =========================

@mcp.custom_route("/twilio/voice", methods=["GET"])
async def twilio_voice(request: Request):
    call_sid = request.query_params.get("CallSid")

    vr = VoiceResponse()

    if not call_sid:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    greeting = "Hey. Go ahead."

    audio = elevenlabs_tts(greeting)
    turn_id = "greet"

    write_audio(call_sid, turn_id, audio)

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        timeout=5,
        speech_timeout=2,
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    form = await request.form()

    log("TWILIO RAW FORM PAYLOAD")
    for k, v in form.items():
        print(f"{k} = {repr(v)}", flush=True)

    call_sid = form.get("CallSid")
    user_text = (form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if not call_sid or call_sid not in CALL_CONTEXT:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    ctx = CALL_CONTEXT[call_sid]

    if now() - ctx["start_time"] > MAX_CALL_DURATION_SECONDS:
        goodbye = "Alright, I have to run. Talk soon."
        audio = elevenlabs_tts(goodbye)
        turn_id = str(int(now() * 1000))
        write_audio(call_sid, turn_id, audio)
        vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    if not user_text:
        gather = Gather(
            input="speech",
            action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
            method="POST",
            timeout=5,
            speech_timeout=2,
        )
        vr.append(gather)
        return PlainTextResponse(str(vr), media_type="text/xml")

    ctx["history"].append(("user", user_text))

    reply = openai_generate(call_sid, user_text)
    ctx["history"].append(("assistant", reply))

    audio = elevenlabs_tts(reply)
    turn_id = str(int(now() * 1000))
    write_audio(call_sid, turn_id, audio)

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        timeout=5,
        speech_timeout=2,
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{call_sid}/{turn_id}", methods=["GET"])
async def twilio_audio(request: Request):
    call_sid = request.path_params["call_sid"]
    turn_id = request.path_params["turn_id"]

    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"

    log(f"[CALL {call_sid}] AUDIO FETCH {turn_id}")
    print(f"Exists: {path.exists()}", flush=True)

    if not path.exists():
        return PlainTextResponse("not found", status_code=404)

    return Response(path.read_bytes(), media_type="audio/mpeg")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
    )
