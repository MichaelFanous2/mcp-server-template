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
# CLIENTS
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

def gpt_generate_text(user_text: str) -> str:
    """
    Uses OpenAI Responses API with GPT-4.1-mini (latest).
    """
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4.1-mini",
            "input": user_text,
            "instructions": (
                "You are a sharp, natural conversational partner on a phone call. "
                "Respond clearly and concisely. Sound human. "
                "Do not mention being an AI. Do not ramble."
            ),
            "max_output_tokens": 120,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["output"][0]["content"][0]["text"].strip()


def elevenlabs_tts_mp3(text: str) -> bytes:
    """
    ElevenLabs highest-quality model.
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


def write_audio(call_sid: str, turn_id: str, mp3: bytes) -> Path:
    path = AUDIO_DIR / f"{call_sid}_{turn_id}.mp3"
    with open(path, "wb") as f:
        f.write(mp3)
    return path


def new_turn_id() -> str:
    return str(int(time.time() * 1000))


# ======================
# MCP TOOL
# ======================

@mcp.tool(description="Start an AI phone call (GPT-4.1-mini + ElevenLabs)")
def start_agent_call(to: str) -> str:
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=f"{PUBLIC_BASE_URL}/twilio/voice",
        method="GET",
    )
    return f"Call started ({call.sid})"


# ======================
# TWILIO WEBHOOKS
# ======================

@mcp.custom_route("/twilio/voice", methods=["GET"])
async def twilio_voice(request: Request):
    call_sid = request.query_params.get("CallSid")

    vr = VoiceResponse()

    if call_sid:
        greeting = "Hey. Go ahead."
        mp3 = elevenlabs_tts_mp3(greeting)
        turn_id = "greet_" + new_turn_id()
        write_audio(call_sid, turn_id, mp3)

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


@mcp.custom_route("/twilio/voice/handle", methods=["POST"])
async def twilio_voice_handle(request: Request):
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()
    call_sid = (form.get("CallSid") or "").strip()

    vr = VoiceResponse()

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

    reply_text = gpt_generate_text(user_text)
    mp3 = elevenlabs_tts_mp3(reply_text)
    turn_id = new_turn_id()
    write_audio(call_sid, turn_id, mp3)

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
