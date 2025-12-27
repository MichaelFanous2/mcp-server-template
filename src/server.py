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
# LOGGING
# =========================
def log(msg: str):
    print(f"\n=== {msg} ===\n", flush=True)


def now() -> float:
    return time.time()


def audio_path(call_sid: str, turn_id: str) -> Path:
    return AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"


def write_audio(call_sid: str, turn_id: str, audio: bytes) -> Path:
    p = audio_path(call_sid, turn_id)
    with open(p, "wb") as f:
        f.write(audio)
    return p


# =========================
# OPENAI (text only)
# =========================
def openai_generate(call_sid: str, user_text: str) -> str:
    ctx = CALL_CONTEXT[call_sid]
    topic = ctx["topic"]
    history: List[Tuple[str, str]] = ctx["history"]

    transcript = ""
    for role, text in history:
        transcript += f"{role.upper()}: {text}\n"

    instructions = (
        "You are in a live phone conversation.\n"
        "Be natural, helpful, concise, and human.\n"
        "Do NOT mention AI, tools, or systems.\n"
        f"Topic/context: {topic}\n"
    )

    payload = {
        "model": "gpt-4.1-mini",
        "instructions": instructions,
        "input": transcript + f"USER: {user_text}\nASSISTANT:",
        "max_output_tokens": 220,
    }

    log(f"[CALL {call_sid}] OPENAI REQUEST (FULL)")
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

    log(f"[CALL {call_sid}] OPENAI RAW RESPONSE (FULL)")
    print(resp.text, flush=True)

    resp.raise_for_status()
    data = resp.json()

    # Standard Responses API structure
    text = data["output"][0]["content"][0]["text"]
    return (text or "").strip()


# =========================
# ELEVENLABS (tts only)
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

    log("ELEVENLABS REQUEST (FULL)")
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

    log("ELEVENLABS RAW RESPONSE HEADERS (FULL)")
    print(dict(resp.headers), flush=True)

    resp.raise_for_status()
    audio = resp.content

    log(f"ELEVENLABS AUDIO BYTES = {len(audio)}")
    return audio


# =========================
# MCP TOOLS
# =========================
@mcp.tool(description="Send an SMS via Twilio")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        body=body
    )
    return f"SMS sent ({msg.sid})"


@mcp.tool(description="Start an AI phone call (Twilio places call, ElevenLabs speaks, OpenAI reasons)")
def start_agent_call(to: str, topic: str) -> str:
    """
    IMPORTANT: We create the outbound call using INLINE TWIML (old reliable behavior),
    not using the `url=` webhook method. This avoids call setup failures when Twilio
    can't fetch your webhook during call creation.

    The call audio is STILL ElevenLabs (played via <Play>).
    The back-and-forth uses Gather -> /twilio/voice/handle.
    """
    log("ABOUT TO CALL TWILIO (INLINE TWIML)")
    print(f"TO   = {to}", flush=True)
    print(f"FROM = {TWILIO_PHONE_NUMBER}", flush=True)
    print(f"BASE = {PUBLIC_BASE_URL}", flush=True)

    # We don't have a CallSid until after Twilio creates the call.
    # So we generate greeting audio AFTER we get call.sid.
    # But we must return TwiML immediately. Solution:
    # - Create call with a short placeholder Gather that hits /twilio/voice/handle with empty speech,
    #   BUT we want greeting first. Better:
    # - Use a predictable temporary ID for greeting (pre_sid), then rename after.
    #
    # Instead, simplest reliable method:
    # 1) Create call with TwiML that immediately fetches greeting audio at /twilio/audio/pending/<pending_id>
    # 2) Store pending context keyed by pending_id
    # 3) When Twilio hits /twilio/audio/pending/<pending_id>, we synthesize greeting, then serve it.
    #
    # That allows call creation to be fully inline and still play ElevenLabs first.

    pending_id = str(int(now() * 1000))
    greeting_text = "Hey. Go ahead."

    # Store pending context; once we learn CallSid from Twilio webhooks, we bind it.
    CALL_CONTEXT[f"PENDING:{pending_id}"] = {
        "topic": topic,
        "history": [],
        "start_time": now(),
        "pending_id": pending_id,
        "greeting_text": greeting_text,
        "bound_call_sid": None,
    }

    # Inline TwiML: Gather will play greeting audio then listen.
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle?pending_id={pending_id}",
        method="POST",
        barge_in=True,
        timeout=5,
        speech_timeout=2,
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/pending/{pending_id}")
    vr.append(gather)

    twiml = str(vr)

    log("INLINE TWIML USED FOR CALL CREATION (FULL)")
    print(twiml, flush=True)

    try:
        call = twilio.calls.create(
            to=to,
            from_=TWILIO_PHONE_NUMBER,
            twiml=twiml
        )
    except Exception as e:
        log("TWILIO CALL CREATE FAILED (FULL)")
        print(repr(e), flush=True)
        raise

    log("TWILIO CALL CREATED")
    print(f"CALL SID = {call.sid}", flush=True)

    # Bind pending context to real call sid
    pending_key = f"PENDING:{pending_id}"
    ctx = CALL_CONTEXT.get(pending_key)
    if ctx:
        ctx["bound_call_sid"] = call.sid
        CALL_CONTEXT[call.sid] = {
            "topic": topic,
            "history": [],
            "start_time": ctx["start_time"],
            "pending_id": pending_id,
        }

    return f"Call started ({call.sid})"


# =========================
# ROUTES (FastMCP custom routes)
# =========================
@mcp.custom_route("/twilio/audio/pending/{pending_id}", methods=["GET"])
async def twilio_audio_pending(request: Request):
    pending_id = request.path_params["pending_id"]
    pending_key = f"PENDING:{pending_id}"
    ctx = CALL_CONTEXT.get(pending_key)

    log(f"[PENDING {pending_id}] AUDIO FETCH (PENDING GREETING)")

    if not ctx:
        return PlainTextResponse("pending not found", status_code=404)

    # Synthesize greeting on demand
    greeting_text = ctx.get("greeting_text") or "Hey. Go ahead."
    audio = elevenlabs_tts(greeting_text)

    # If the call sid is already known, store it under call_sid as well, but we can serve immediately.
    # For serving this request, we just return bytes.
    log(f"[PENDING {pending_id}] SERVING GREETING AUDIO BYTES = {len(audio)}")
    return Response(audio, media_type="audio/mpeg")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    # pending_id might be in query params for the first turn
    pending_id = request.query_params.get("pending_id")

    form = await request.form()
    log("TWILIO RAW FORM PAYLOAD (FULL)")
    for k, v in form.items():
        print(f"{k} = {repr(v)}", flush=True)

    call_sid = form.get("CallSid")
    user_text = (form.get("SpeechResult") or "").strip()

    log("TWILIO PARSED FIELDS")
    print(f"pending_id = {repr(pending_id)}", flush=True)
    print(f"call_sid    = {repr(call_sid)}", flush=True)
    print(f"user_text   = {repr(user_text)}", flush=True)

    vr = VoiceResponse()

    if not call_sid:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    # Ensure we have context
    if call_sid not in CALL_CONTEXT:
        # Try to bind from pending
        if pending_id:
            pending_key = f"PENDING:{pending_id}"
            pctx = CALL_CONTEXT.get(pending_key)
            if pctx:
                CALL_CONTEXT[call_sid] = {
                    "topic": pctx["topic"],
                    "history": [],
                    "start_time": pctx["start_time"],
                    "pending_id": pending_id,
                }

    if call_sid not in CALL_CONTEXT:
        log(f"[CALL {call_sid}] NO CONTEXT FOUND -> HANGUP")
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    ctx = CALL_CONTEXT[call_sid]

    # Hard timeout
    elapsed = now() - ctx["start_time"]
    log(f"[CALL {call_sid}] ELAPSED SECONDS = {elapsed}")
    if elapsed > MAX_CALL_DURATION_SECONDS:
        goodbye = "Alright, I have to run. Talk soon."
        audio = elevenlabs_tts(goodbye)
        turn_id = str(int(now() * 1000))
        write_audio(call_sid, turn_id, audio)
        vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    # If empty speech, just re-arm gather
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

    # Append user message
    ctx["history"].append(("user", user_text))

    # OpenAI
    reply = openai_generate(call_sid, user_text)
    ctx["history"].append(("assistant", reply))

    # ElevenLabs
    audio = elevenlabs_tts(reply)
    turn_id = str(int(now() * 1000))
    write_audio(call_sid, turn_id, audio)

    # Respond with Gather+Play so the loop continues
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

    p = audio_path(call_sid, turn_id)

    log(f"[CALL {call_sid}] AUDIO FETCH turn_id={turn_id}")
    print(f"path={str(p)} exists={p.exists()}", flush=True)
    if p.exists():
        print(f"size={p.stat().st_size}", flush=True)

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
