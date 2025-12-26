#!/usr/bin/env python3
import os
import base64
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
# CLIENTS
# ======================

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
mcp = FastMCP("Twilio MCP")


# ======================
# IN-MEMORY AUDIO CACHE
# ======================

AUDIO_CACHE = {}  # CallSid -> base64 mp3


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


def elevenlabs_tts(text: str) -> str:
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
        raise RuntimeError(resp.text)

    return base64.b64encode(resp.content).decode("utf-8")


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


@mcp.tool(description="Make a static phone call via Twilio")
def make_call(to: str, message: str) -> str:
    vr = VoiceResponse()
    vr.say(message, voice="alice")

    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(vr),
    )
    return f"Call started ({call.sid})"


@mcp.tool(description="Start an interactive AI phone call (Twilio + Poke + ElevenLabs)")
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
    vr.say(greeting, voice="alice")

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        speech_timeout="auto",
    )
    gather.say("Tell me what you need.", voice="alice")
    vr.append(gather)

    vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()
    call_sid = (form.get("CallSid") or "").strip()

    vr = VoiceResponse()

    if not user_text:
        vr.say("I did not catch that. Can you repeat?", voice="alice")
        vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
        return PlainTextResponse(str(vr), media_type="text/xml")

    reply = poke_reply(user_text)
    audio_b64 = elevenlabs_tts(reply)

    if call_sid:
        AUDIO_CACHE[call_sid] = audio_b64
        vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}")
    else:
        vr.say(reply, voice="alice")

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        speech_timeout="auto",
    )
    vr.append(gather)

    return PlainTextResponse(str(vr), media_type="text/xml")


@mcp.custom_route("/twilio/audio/{call_sid}", methods=["GET"])
async def twilio_audio(request: Request):
    call_sid = request.path_params.get("call_sid")
    b64 = AUDIO_CACHE.get(call_sid)

    if not b64:
        return PlainTextResponse("not found", status_code=404)

    audio_bytes = base64.b64decode(b64.encode("utf-8"))
    return Response(audio_bytes, media_type="audio/mpeg")


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
