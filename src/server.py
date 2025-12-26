#!/usr/bin/env python3
import os
import base64
import requests
from fastapi import FastAPI, Request
from fastmcp import FastMCP
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather

# -----------------------
# Env
# -----------------------
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

POKE_API_KEY = os.environ["POKE_API_KEY"]
POKE_INBOUND_URL = os.environ.get(
    "POKE_INBOUND_URL",
    "https://poke.com/api/v1/inbound-sms/webhook"
)

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]  # required, no default

# -----------------------
# Clients
# -----------------------
twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -----------------------
# MCP server (tools)
# -----------------------
mcp = FastMCP("Twilio MCP")

@mcp.tool(description="Send an SMS via Twilio")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        body=body
    )
    return f"SMS sent ({msg.sid})"

@mcp.tool(description="Start an AI phone call powered by Poke + ElevenLabs (interactive loop)")
def start_agent_call(to: str, greeting: str = "Hey, how can I help?") -> str:
    """
    Starts a call that hits our Twilio webhook which runs:
    user speech -> Poke -> ElevenLabs -> play -> gather -> repeat
    """
    url = f"{PUBLIC_BASE_URL}/twilio/voice?greeting={requests.utils.quote(greeting)}"
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        url=url,  # Twilio will GET this to fetch TwiML
        method="GET",
    )
    return f"Call started ({call.sid})"

# Optional: keep your old make_call tool if you still want it
@mcp.tool(description="Make a phone call via Twilio and say a static message (non-interactive)")
def make_call(to: str, message: str) -> str:
    response = VoiceResponse()
    response.say(message, voice="alice")
    call = twilio.calls.create(
        to=to,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(response),
    )
    return f"Call started ({call.sid})"

# -----------------------
# Helper: talk to Poke
# -----------------------
def poke_generate_reply(user_text: str) -> str:
    resp = requests.post(
        POKE_INBOUND_URL,
        headers={
            "Authorization": f"Bearer {POKE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"message": user_text},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    # Best-effort parsing since we do not have the exact schema guarantee here
    # Common patterns: {"message": "..."} or {"reply": "..."} or {"output": "..."}
    for key in ["message", "reply", "output", "text"]:
        if isinstance(data, dict) and isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()

    # If schema differs, at least fail loudly with context
    raise RuntimeError(f"Unexpected Poke response schema: {data}")

# -----------------------
# Helper: ElevenLabs TTS
# -----------------------
def elevenlabs_tts_to_base64_mp3(text: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text}")
    return base64.b64encode(resp.content).decode("utf-8")

# -----------------------
# HTTP app for Twilio webhooks
# -----------------------
# FastMCP runs an HTTP server. In many setups it exposes a FastAPI app object.
# If your fastmcp version exposes `mcp.app`, use it. Otherwise, see note below.
app: FastAPI = getattr(mcp, "app", FastAPI())

@app.get("/twilio/voice")
async def twilio_voice(greeting: str = "Hey, how can I help?"):
    """
    First webhook Twilio hits. Return TwiML that greets and gathers speech.
    """
    vr = VoiceResponse()
    vr.say(greeting, voice="alice")

    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        speech_timeout="auto"
    )
    gather.say("Tell me what you need.", voice="alice")
    vr.append(gather)

    # If no speech captured, loop
    vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
    return str(vr)

@app.post("/twilio/voice/handle")
async def twilio_voice_handle(request: Request):
    """
    Receives user's speech. Sends to Poke. Converts reply to ElevenLabs audio.
    Plays it. Then gathers again.
    """
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if not user_text:
        vr.say("I did not catch that. Can you repeat?", voice="alice")
        vr.redirect(f"{PUBLIC_BASE_URL}/twilio/voice", method="GET")
        return str(vr)

    # 1) Poke generates response text
    reply_text = poke_generate_reply(user_text)

    # 2) ElevenLabs generates audio
    audio_b64 = elevenlabs_tts_to_base64_mp3(reply_text)

    # 3) Play the audio to the caller
    # Twilio cannot directly Play base64. We need a URL.
    # Easiest approach: return the text with <Say> OR host the audio bytes.
    # We'll host the audio bytes via a short-lived endpoint keyed by CallSid.
    call_sid = str(form.get("CallSid") or "")
    if not call_sid:
        # fallback: just say it with Twilio if we cannot associate call
        vr.say(reply_text, voice="alice")
    else:
        # Store audio in memory for this call. This is fine for small scale.
        AUDIO_CACHE[call_sid] = audio_b64
        vr.play(f"{PUBLIC_BASE_URL}/twilio/audio/{call_sid}")

    # 4) Gather again
    gather = Gather(
        input="speech",
        action=f"{PUBLIC_BASE_URL}/twilio/voice/handle",
        method="POST",
        speech_timeout="auto"
    )
    vr.append(gather)

    return str(vr)

# In-memory audio cache for demo. For production, use Redis/S3.
AUDIO_CACHE = {}

@app.get("/twilio/audio/{call_sid}")
async def twilio_audio(call_sid: str):
    """
    Serves the MP3 bytes Twilio will fetch and play.
    """
    b64 = AUDIO_CACHE.get(call_sid)
    if not b64:
        # Twilio expects audio. If missing, return silence-ish error.
        return ("", 404, {"Content-Type": "text/plain"})

    audio_bytes = base64.b64decode(b64.encode("utf-8"))

    # Optionally delete after first fetch so it does not grow unbounded.
    # If Twilio retries, this might break playback. You can keep it for a few minutes instead.
    # del AUDIO_CACHE[call_sid]

    return (audio_bytes, 200, {"Content-Type": "audio/mpeg"})

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True
    )
