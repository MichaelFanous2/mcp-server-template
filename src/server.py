#!/usr/bin/env python3
import os
import time
from pathlib import Path
import requests

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather


# ======================
# REQUIRED ENV VARS
# ======================
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

POKE_API_KEY = os.environ["POKE_API_KEY"]
POKE_INBOUND_URL = os.environ.get(
    "POKE_INBOUND_URL",
    "https://poke.com/api/v1/inbound-sms/webhook",
)

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]


# ======================
# CLIENTS + SERVER
# ======================
twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# ======================
# AUDIO STORAGE
# ======================
AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# HELPERS
# ======================
def poke_generate_text(user_text: str) -> str:
    """
    WARNING: This endpoint is Poke's inbound SMS webhook.
    It returns text, but may also create SMS-side artifacts in Poke depending on their system.
    If Poke offers a non-SMS inference endpoint, swap it in here.
    """
    resp = requests.post(
        POKE_INBOUND_URL,
        headers={
            "Authorization": f"Bearer {POKE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"message": user_text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    for key in ("message", "reply", "output", "text"):
        val = data.get(key) if isinstance(data, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()

    raise RuntimeError(f"Unexpected Poke response: {data}")


def elevenlabs_tts_mp3_bytes(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text}")
    return resp.content


def write_audio(call_sid: str, turn_id: str, mp3_bytes: bytes) -> Path:
    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    with open(path, "wb") as f:
        f.write(mp3_bytes)
    return path


def new_turn_id() -> str:
    return str(int(time.time() * 1000))


# ======================
# MCP TOOLS
# ======================
@mcp.tool(description="Send an SMS via Twilio")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(to=to, from_=TWILIO_PHONE_NUMBER, body=body)
    return f"SMS sent ({msg.sid})"


@mcp.tool(description="Start an interactive call: Twilio calls, ElevenLabs is the only voice")
def start_agent_call(to: str) -> str:
    # Twilio places the call; Twilio hits our webhook; our webhook only uses <Play> with ElevenLabs MP3.
    url = f"{PUBLIC_BASE_URL}/twilio/voice"
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=url,
        method="GET",
    )
    return f"Call started ({call.sid})"


# ======================
# TWILIO WEBHOOKS (NO <Say> ANYWHERE)
# ======================
@mcp.custom_route("/twilio/voice", methods=["GET"])
async def twilio_voice(request: Request):
    # Twilio includes CallSid in query params on webhook requests.
    call_sid = (request.query_params.get("CallSid") or "").strip()

    vr = VoiceResponse()

    # Generate a short greeting in ElevenLabs. No Twilio voice, ever.
    greeting_text = "Hey. Talk to me."
    turn_id = "greet_" + new_turn_id()

    if call_sid:
        mp3 = elevenlabs_tts_mp3_bytes(greeting_text)
        write_audio(call_sid, turn_id, mp3)
        audio_url = f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}"
    else:
        # If for some reason no CallSid, we still won't use <Say>.
        # Return a gather that immediately listens.
        audio_url = None

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )

    # Put <Play> INSIDE <Gather> so user can interrupt the greeting.
    if audio_url:
        gather.play(audio_url)

    vr.append(gather)

    # If nothing was captured, loop.
    vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()
    call_sid = (form.get("CallSid") or "").strip()

    if not call_sid:
        # Hard fail: we refuse to ever speak via Twilio.
        raise RuntimeError("Missing CallSid; refusing to fall back to Twilio voice")

    vr = VoiceResponse()

    if not user_text:
        # Re-listen, no Twilio voice prompt.
        gather = Gather(
            input="speech",
            action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
            method="POST",
            barge_in=True,
            speech_timeout="auto",
        )
        vr.append(gather)
        return PlainTextResponse(str(vr), media_type="text/xml")

    # 1) Poke generates text
    reply_text = poke_generate_text(user_text)

    # 2) ElevenLabs generates MP3 (this is the only voice)
    turn_id = new_turn_id()
    mp3 = elevenlabs_tts_mp3_bytes(reply_text)
    write_audio(call_sid, turn_id, mp3)

    audio_url = f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}/{turn_id}"

    # 3) Play audio INSIDE Gather so user can interrupt and talk naturally
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )
    gather.play(audio_url)
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{call_sid}/{turn_id}", methods=["GET"])
async def twilio_audio(request: Request):
    call_sid = request.path_params.get("call_sid")
    turn_id = request.path_params.get("turn_id")

    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    if not path.exists():
        # If this happens, Twilio will play nothing (still no <Say> fallback).
        return PlainTextResponse("audio not found", status_code=404)

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
