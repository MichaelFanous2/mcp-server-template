#!/usr/bin/env python3
import os
import base64
import requests
from pathlib import Path

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
# CLIENTS
# ======================

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# ======================
# AUDIO STORAGE (DURABLE)
# ======================

AUDIO_DIR = Path("/tmp/elevenlabs_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# HELPERS
# ======================

def poke_reply(user_text: str) -> str:
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
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    raise RuntimeError(f"Unexpected Poke response: {data}")


def elevenlabs_tts_to_mp3(text: str, call_sid: str) -> Path:
    """
    Generates ElevenLabs audio and writes it to disk.
    Returns the file path. Raises if anything fails.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text}")

    audio_path = AUDIO_DIR / f"{call_sid}.mp3"
    with open(audio_path, "wb") as f:
        f.write(resp.content)

    return audio_path


# ======================
# MCP TOOLS
# ======================

@mcp.tool(description="Send an SMS via Twilio")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        body=body,
    )
    return f"SMS sent ({msg.sid})"


@mcp.tool(description="Start an interactive AI phone call (Twilio + Poke + ElevenLabs ONLY)")
def start_agent_call(to: str, greeting: str = "Hey, how can I help?") -> str:
    url = f"{PUBLIC_BASE_URL}/twilio/voice?greeting={requests.utils.quote(greeting)}"
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=url,
        method="GET",
    )
    return f"Call started ({call.sid})"


# ======================
# TWILIO WEBHOOKS
# ======================

@mcp.custom_route("/twilio/voice", methods=["GET"])
async def twilio_voice(request: Request):
    greeting = request.query_params.get("greeting") or "Hey, how can I help?"

    vr = VoiceResponse()

    # IMPORTANT: no Twilio <Say> for normal flow.
    # We only gather speech here.
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )
    gather.say(greeting, voice="alice")  # greeting only; conversation uses ElevenLabs
    vr.append(gather)

    vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()
    call_sid = (form.get("CallSid") or "").strip()

    vr = VoiceResponse()

    if not call_sid:
        # Hard fail: never fall back to Twilio voice
        raise RuntimeError("Missing CallSid; refusing to use Twilio TTS")

    if not user_text:
        # Re-prompt, but still no Alice conversation
        gather = Gather(
            input="speech",
            action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
            method="POST",
            barge_in=True,
            speech_timeout="auto",
        )
        vr.append(gather)
        return PlainTextResponse(str(vr), media_type="text/xml")

    # 1) User speech -> Poke
    reply = poke_reply(user_text)

    # 2) Poke reply -> ElevenLabs (durable audio)
    audio_path = elevenlabs_tts_to_mp3(reply, call_sid)

    # 3) Twilio plays ElevenLabs audio (no fallback)
    vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}")

    # 4) Continue the loop
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        barge_in=True,
        speech_timeout="auto",
    )
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{call_sid}", methods=["GET"])
async def twilio_audio(request: Request):
    call_sid = request.path_params.get("call_sid")
    audio_path = AUDIO_DIR / f"{call_sid}.mp3"

    if not audio_path.exists():
        # If this 404s, Twilio will not speak at all.
        # That is intentional. No Alice fallback.
        return PlainTextResponse("audio not found", status_code=404)

    with open(audio_path, "rb") as f:
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
