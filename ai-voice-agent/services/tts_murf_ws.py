# day 20
# services/tts_murf_ws.py
import os
import json
import aiohttp

MURF_API_KEY = os.getenv("MURF_API_KEY")
WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

async def stream_tts_with_murf_ws(
    text: str,
    websocket=None,
    *,
    voice_id: str = "en-US-amara",
    sample_rate: int = 44100,
    channel_type: str = "MONO",
    audio_format: str = "WAV",
    context_id: str = "day20-static-context"  # static to avoid context limit errors
) -> None:
    """
    Send final LLM text to Murf via WebSocket, receive audio (base64 WAV),
    and print base64 chunks to server console. No UI changes.
    """
    if not MURF_API_KEY:
        print("❗ MURF_API_KEY missing")
        return

    qs = f"?api-key={MURF_API_KEY}&sample_rate={sample_rate}&channel_type={channel_type}&format={audio_format}"
    url = f"{WS_URL}{qs}"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            # Optional: send a static context id first (prevents context churn)
            # If Murf ignores unknown messages, this is harmless.
            await ws.send_str(json.dumps({"context_id": context_id}))

            # Voice configuration first
            voice_config_msg = {
                "voice_config": {
                    "voiceId": voice_id,
                    "style": "Conversational",
                    "rate": 0,
                    "pitch": 0,
                    "variation": 1
                }
            }
            await ws.send_str(json.dumps(voice_config_msg))

            # Send the text (you could chunk if you ever want partial TTS)
            text_msg = {
                "text": text,
                "end": True  # finalize
            }
            await ws.send_str(json.dumps(text_msg))

            print("🎤 Sent text to Murf. Streaming base64 audio chunks below…")

            # Receive base64 audio chunks; print them to console
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        print("⚠️ Non-JSON message from Murf:", msg.data)
                        continue

                    if "audio" in data:
                        audio_b64 = data["audio"]
                        # Print full base64 (screenshot this for LinkedIn)
                        print(f"\n🎵 Murf Audio (chunk len={len(audio_b64)}): {audio_b64[:100]}...")
                        # ✅ Send chunk to client if websocket is provided
                        if websocket:
                            await websocket.send_text(json.dumps({
                                "type": "audio_chunk",
                                "chunk": audio_b64
                            }))
                            print("✅ Audio chunk forwarded to client")

                    if data.get("final"):
                        print("\n✅ Murf TTS completed.")
                        break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print("❌ Murf WebSocket error")
                    break
