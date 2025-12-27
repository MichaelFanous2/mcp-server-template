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
# ENV
# =========================
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]


# =========================
# CLIENTS / SERVER
# =========================
twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# =========================
# STATE
# =========================
AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CALL_CONTEXT: Dict[str, Dict[str, Any]] = {}
MAX_CALL_DURATION_SECONDS = 240


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
    return f"Hey. Let’s talk about {topic}. Go ahead."


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

    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(vr),
        record=True,
        recording_channels="dual",
    )

    CALL_CONTEXT[call.sid] = {
        "topic": topic,
        "history": [],
        "start_time": now(),
        "pending_id": pending_id,
    }

    return f"Call started ({call.sid})"


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
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
    )
