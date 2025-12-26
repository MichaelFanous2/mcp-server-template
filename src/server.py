#!/usr/bin/env python3
import os
from fastmcp import FastMCP
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse


# MCP server
mcp = FastMCP("Twilio MCP")

# Twilio client
twilio = Client(
    os.environ["TWILIO_ACCOUNT_SID"],
    os.environ["TWILIO_AUTH_TOKEN"]
)

@mcp.tool(description="Send an SMS via Twilio")
def send_sms(to: str, body: str) -> str:
    msg = twilio.messages.create(
        to=to,
        from_=os.environ["TWILIO_PHONE_NUMBER"],
        body=body
    )
    return f"SMS sent ({msg.sid})"

@mcp.tool(description="Make a phone call via Twilio and say a message")
def make_call(to: str, message: str) -> str:
    response = VoiceResponse()
    response.say(message, voice="alice")

    call = twilio.calls.create(
        to=to,
        from_=os.environ["TWILIO_PHONE_NUMBER"],
        twiml=str(response),
    )

    return f"Call started ({call.sid})"

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True
    )
