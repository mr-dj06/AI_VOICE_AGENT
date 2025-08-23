import asyncio
import websockets
import json
import base64
import os

MURF_API_KEY = os.getenv("MURF_API_KEY")
WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

async def stream_tts_with_murf(text: str):
    """
    Streams text to Murf via WebSocket for TTS and prints base64 audio chunks.
    """
    async with websockets.connect(
        f"{WS_URL}?api-key={MURF_API_KEY}&sample_rate=44100&channel_type=MONO&format=WAV"
    ) as ws:
        
        # ✅ Send voice configuration
        voice_config_msg = {
            "voice_config": {
                "voiceId": "en-US-amara",  # Change if needed
                "style": "Conversational",
                "rate": 0,
                "pitch": 0,
                "variation": 1
            }
        }
        await ws.send(json.dumps(voice_config_msg))
        
        # ✅ Send text input (you can chunk this if needed)
        text_msg = {
            "text": text,
            "end": True  # Marks end of input so Murf knows to finalize
        }
        await ws.send(json.dumps(text_msg))
        
        print("🎤 Sent text to Murf, waiting for audio chunks...")
        
        first_chunk = True
        audio_base64_combined = ""
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if "audio" in data:
                # ✅ Print base64 audio (this is what you need for Day 20)
                audio_base64 = data["audio"]
                audio_base64_combined += audio_base64
                print("\n🎵 Audio Chunk (base64):")
                print(audio_base64[:100] + "...")  # Print first 100 chars for brevity
                
            if data.get("final"):
                print("\n✅ Murf TTS Completed")
                break
        
        return audio_base64_combined
