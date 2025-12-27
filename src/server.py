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


# ======================
# ENV VARS (REQUIRED)
# ======================

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]


# ======================
# CLIENTS + MCP
# ======================

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# ======================
# AUDIO STORAGE
# ======================

AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# CALL CONTEXT (IN-MEMORY)
# call_sid -> {
#   "topic": str,
#   "history": List[Tuple[str, str]],  # ("user"|"assistant", text)
#   "created_ms": int
# }
# ======================

CALL_CONTEXT: Dict[str, Dict[str, Any]] = {}
CONTEXT_TTL_MS = 60 * 60 * 1000  # 1 hour


def _now_ms() -> int:
    return int(time.time() * 1000)


def _gc_context() -> None:
    """Best-effort cleanup to avoid unbounded growth."""
    cutoff = _now_ms() - CONTEXT_TTL_MS
    dead = [sid for sid, ctx in CALL_CONTEXT.items() if ctx.get("created_ms", 0) < cutoff]
    for sid in dead:
        CALL_CONTEXT.pop(sid, None)


def _new_turn_id() -> str:
    return str(_now_ms())


def _write_audio(call_sid: str, turn_id: str, mp3: bytes) -> Path:
    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    with open(path, "wb") as f:
        f.write(mp3)
    return path


def _openai_generate_text(call_sid: str, user_text: str) -> str:
    """
    Uses OpenAI Responses API with gpt-4.1-mini.
    We inject the topic + recent history to keep the conversation coherent.
    """
    ctx = CALL_CONTEXT.get(call_sid) or {}
    topic = ctx.get("topic", "general conversation")
    history: List[Tuple[str, str]] = ctx.get("history", [])

    # Keep last N turns to control token usage
    last_turns = history[-10:]

    # Build a compact transcript
    transcript_lines = []
    for role, text in last_turns:
        if not text:
            continue
        if role == "user":
            transcript_lines.append(f"User: {text}")
        else:
            transcript_lines.append(f"Assistant: {text}")

    transcript = "\n".join(transcript_lines).strip()

    instructions = (
        "You are a sharp, natural conversational partner on a phone call. "
        "You speak concisely and confidently. "
        "Do not mention being an AI, a model, OpenAI, or any system details. "
        "No meta talk about tools or integrations.\n\n"
        f"Topic anchor for this call: {topic}\n\n"
        "If the user changes the subject, follow them naturally.\n"
    )

    # The user's latest utterance comes last
    if transcript:
        input_text = f"{transcript}\nUser: {user_text}\nAssistant:"
    else:
        input_text = f"User: {user_text}\nAssistant:"

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4.1-mini",
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": 160,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Typical Responses API shape
    try:
        text = data["output"][0]["content"][0]["text"]
    except Exception:
        # Fallback if response shape changes slightly
        text = str(data)

    return (text or "").strip()


def _elevenlabs_tts_mp3(text: str) -> bytes:
    """
    ElevenLabs TTS using the requested best-quality model.
    """
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def _append_history(call_sid: str, role: str, text: str) -> None:
    if not call_sid:
        return
    ctx = CALL_CONTEXT.setdefault(
        call_sid,
        {"topic": "general conversation", "history": [], "created_ms": _now_ms()},
    )
    ctx["history"].append((role, text))


# ======================
# MCP TOOLS
# ======================

@mcp.tool(description="Start an AI phone call (Twilio call control, OpenAI brain, ElevenLabs voice).")
def start_agent_call(to: str, topic: str = "general conversation") -> str:
    """
    Poke calls THIS tool with (to, topic).
    The topic is stored only on our server, keyed by call.sid.
    Twilio never sees or needs the topic.
    """
    _gc_context()

    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=f"{PUBLIC_BASE_URL}/twilio/voice",
        method="GET",
    )

    CALL_CONTEXT[call.sid] = {
        "topic": topic or "general conversation",
        "history": [],
        "created_ms": _now_ms(),
    }

    return f"Call started ({call.sid})"


@mcp.tool(description="Send an SMS via Twilio (optional utility).")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        body=body,
    )
    return f"SMS sent ({msg.sid})"


# ======================
# TWILIO WEBHOOKS
# ======================

@mcp.custom_route("/twilio/voice", methods=["GET"])
async def twilio_voice(request: Request):
    """
    Initial TwiML for the call.
    We play an ElevenLabs greeting INSIDE Gather so the user can barge in.
    No <Say>. No Twilio voice.
    """
    _gc_context()

    call_sid = (request.query_params.get("CallSid") or "").strip()
    vr = VoiceResponse()

    if not call_sid:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    # Ensure context exists even if someone hits webhook before tool stored it
    ctx = CALL_CONTEXT.setdefault(
        call_sid,
        {"topic": "general conversation", "history": [], "created_ms": _now_ms()},
    )

    greeting = "Hey. Go ahead."
    turn_id = "greet_" + _new_turn_id()

    mp3 = _elevenlabs_tts_mp3(greeting)
    _write_audio(call_sid, turn_id, mp3)

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
    vr.append(gather)

    # If nothing captured, loop back
    vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    """
    Receives Twilio speech transcription, sends it to OpenAI with topic + history,
    then generates ElevenLabs audio and plays it, then gathers again.
    """
    _gc_context()

    form = await request.form()
    call_sid = (form.get("CallSid") or "").strip()
    user_text = (form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if not call_sid:
        vr.hangup()
        return PlainTextResponse(str(vr), media_type="text/xml")

    # If no speech recognized, just gather again silently
    if not user_text:
        gather = Gather(
            input="speech",
            action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
            method="POST",
            barge_in=True,
            speech_timeout="auto",
        )
        vr.append(gather)
        return PlainTextResponse(str(vr), media_type="text/xml")

    _append_history(call_sid, "user", user_text)

    # OpenAI generates the response text using topic + history
    reply_text = _openai_generate_text(call_sid, user_text)
    if not reply_text:
        reply_text = "Got it. Say a bit more."

    _append_history(call_sid, "assistant", reply_text)

    # ElevenLabs generates audio for the reply
    turn_id = _new_turn_id()
    mp3 = _elevenlabs_tts_mp3(reply_text)
    _write_audio(call_sid, turn_id, mp3)

    # Play response inside Gather for barge-in
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )
    gather.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}")
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{call_sid}/{turn_id}", methods=["GET"])
async def twilio_audio(request: Request):
    """
    Twilio fetches the MP3 here to play it to the user.
    """
    call_sid = request.path_params["call_sid"]
    turn_id = request.path_params["turn_id"]

    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    if not path.exists():
        return PlainTextResponse("not found", status_code=404)

    with open(path, "rb") as f:
        return Response(f.read(), media_type="audio/mpeg")


# ======================
# ENTRYPOINT
# ======================

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
    )
