import traceback
import os
import tempfile
import time
import requests
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List

from services.tts_murf import generate_tts_with_murf # type: ignore

from fastapi import WebSocket, WebSocketDisconnect
import uuid
from datetime import datetime

import aiohttp
import asyncio

# Load ENV & Configure APIs
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GEMINI_API_KEY or not ASSEMBLY_API_KEY or not MURF_API_KEY:
    raise RuntimeError("❗ Missing one or more API keys in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

# Create audio recordings directory
RECORDINGS_DIR = Path("audio_recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

# Load chat history from JSON file
def load_chat_history() -> Dict[str, List[dict]]:
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save chat history to JSON file
def save_chat_history(history: Dict[str, List[dict]]):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# Initialize chat history
chat_history: Dict[str, List[dict]] = load_chat_history()

# FastAPI App
app = FastAPI(title="30-Days-of-AI-Voice-Agents – Day 16")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    # return FileResponse(static_path / "audio.html") # for day 16 receive audio chunks
    return FileResponse(static_path / "transcribe.html") # for day 17
    # return FileResponse(static_path / "index.html") # default file response

FALLBACK_MESSAGE="I'm having trouble connecting right now."

# Core LLM Endpoint (handles both text & audio)
@app.post("/llm/query")
async def query_llm(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Handles either:
    1. Text input
    2. Audio input (transcribed via AssemblyAI)
    """
    try:
        # Case 1: If audio file is provided
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Upload to AssemblyAI
            try:
                headers = {"authorization": ASSEMBLY_API_KEY}
                upload_url = "https://api.assemblyai.com/v2/upload"
                with open(tmp_path, "rb") as f:
                    upload_resp = requests.post(upload_url, headers=headers, data=f)
                audio_url = upload_resp.json()["upload_url"]
            except Exception as e:
                print("upload failed due to=> ", e)
                return {"services": FALLBACK_MESSAGE}

            # Start transcription
            try:
                transcript_url = "https://api.assemblyai.com/v2/transcript"
                transcript_req = {"audio_url": audio_url}
                transcript_resp = requests.post(transcript_url, json=transcript_req, headers=headers)
                transcript_id = transcript_resp.json()["id"]
            except Exception as e:
                print("transcription failed due to=> ",e)
                return {"services": FALLBACK_MESSAGE}

            # Poll until complete
            while True:
                poll_resp = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
                status = poll_resp.json()["status"]
                if status == "completed":
                    text = poll_resp.json()["text"]
                    break
                elif status == "error":
                    raise HTTPException(status_code=500, detail="Transcription failed")
                time.sleep(2)

        if not text:
            raise HTTPException(status_code=400, detail="No text or audio provided")

        # Step 2: Send text to Gemini
        try:
            gemini_resp = gemini_model.generate_content(text)
            ai_text = gemini_resp.text.strip()
        except Exception as e:
            print("error from gemini ai due to => ", e)
            return {"services": FALLBACK_MESSAGE}

        # Step 3: Send AI text to Murf for TTS
        murf_audio_url = generate_tts_with_murf(ai_text)
        return {
            "transcribed_text": text if file else None,
            "ai_text": ai_text,
            "audio_url": murf_audio_url
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/agent/chat/{session_id}")
async def chat_with_history(
    session_id: str,
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Handles conversational chat with history using session_id.
    1. Transcribe audio if provided
    2. Fetch and append to chat history
    3. Query LLM with combined history and new input
    4. Store response in history
    5. Return TTS audio
    """
    try:
        # Initialize chat history for session if not exists
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Step 1: Transcribe audio if provided
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Upload to AssemblyAI
            headers = {"authorization": ASSEMBLY_API_KEY}
            upload_url = "https://api.assemblyai.com/v2/upload"
            with open(tmp_path, "rb") as f:
                upload_resp = requests.post(upload_url, headers=headers, data=f)
            audio_url = upload_resp.json()["upload_url"]

            # Start transcription
            transcript_url = "https://api.assemblyai.com/v2/transcript"
            transcript_req = {"audio_url": audio_url}
            transcript_resp = requests.post(transcript_url, json=transcript_req, headers=headers)
            transcript_id = transcript_resp.json()["id"]

            # Poll until complete
            while True:
                poll_resp = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
                status = poll_resp.json()["status"]
                if status == "completed":
                    text = poll_resp.json()["text"]
                    break
                elif status == "error":
                    raise HTTPException(status_code=500, detail="Transcription failed")
                time.sleep(2)

        if not text:
            raise HTTPException(status_code=400, detail="No text or audio provided")

        # Step 2: Fetch chat history and append new user message
        history = chat_history[session_id]
        history.append({"role": "user", "content": text})
        save_chat_history(chat_history)

        # Step 3: Combine history and new message for LLM
        conversation = ""
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            conversation += f"{role}: {message['content']}\n"
        conversation += "Assistant: "

        # Step 4: Query Gemini with conversation history
        gemini_resp = gemini_model.generate_content(conversation)
        ai_text = gemini_resp.text.strip()

        # Step 5: Append assistant response to history
        history.append({"role": "assistant", "content": ai_text})
        save_chat_history(chat_history)

        # Step 6: Generate TTS for assistant response
        murf_audio_url = generate_tts_with_murf(ai_text)

        return {
            "transcribed_text": text if file else None,
            "ai_text": ai_text,
            "audio_url": murf_audio_url,
            "session_id": session_id
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.get("/agent/history/{session_id}")
async def get_history(session_id: str):
    return chat_history.get(session_id, [])

# Enhanced WebSocket endpoint for streaming audio
@app.websocket("/ws/audio")
async def audio_streaming_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio recording
    1. Accepts WebSocket connection
    2. Receives binary audio data chunks
    3. Saves streaming audio to file
    4. Handles connection lifecycle
    """
    await websocket.accept()
    
    # Generate unique session and file identifiers
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"streaming_audio_{timestamp}_{session_id[:8]}.wav"
    audio_filepath = RECORDINGS_DIR / audio_filename
    
    print(f"🎙️ New audio streaming session started: {session_id}")
    print(f"📁 Audio will be saved to: {audio_filepath}")
    
    # Send initial confirmation to client
    await websocket.send_text(json.dumps({
        "type": "session_started",
        "session_id": session_id,
        "filename": audio_filename,
        "message": "Audio streaming session started"
    }))
    
    audio_file = None
    total_bytes_received = 0
    chunk_count = 0
    
    try:
        # Open file for binary writing
        audio_file = open(audio_filepath, "wb")
        
        while True:
            try:
                # Receive binary audio data
                audio_data = await websocket.receive_bytes()
                chunk_count += 1
                total_bytes_received += len(audio_data)
                
                # Write audio chunk to file
                audio_file.write(audio_data)
                audio_file.flush()  # Ensure data is written immediately
                
                print(f"📦 Chunk {chunk_count}: Received {len(audio_data)} bytes (Total: {total_bytes_received} bytes)")
                
                # Send acknowledgment back to client
                await websocket.send_text(json.dumps({
                    "type": "chunk_received",
                    "chunk_number": chunk_count,
                    "chunk_size": len(audio_data),
                    "total_bytes": total_bytes_received
                }))
                
            except WebSocketDisconnect:
                print(f"🔌 Client disconnected from session {session_id}")
                break
            except Exception as e:
                print(f"❌ Error processing audio chunk: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error processing audio: {str(e)}"
                }))
                break
                
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
    finally:
        # Clean up resources
        if audio_file:
            audio_file.close()
            print(f"💾 Audio file saved: {audio_filepath} ({total_bytes_received} bytes, {chunk_count} chunks)")
        
        # Send final summary to client (if connection still active)
        try:
            await websocket.send_text(json.dumps({
                "type": "session_ended",
                "session_id": session_id,
                "filename": audio_filename,
                "total_bytes": total_bytes_received,
                "total_chunks": chunk_count,
                "message": f"Recording saved successfully with {total_bytes_received} bytes"
            }))
        except:
            pass  # Connection might already be closed
        
        print(f"🏁 Audio streaming session ended: {session_id}")

# Original WebSocket endpoint (kept for compatibility)
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Simple WebSocket Echo Server
    1. Accepts connection
    2. Receives message from client
    3. Echoes message back
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📩 Received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print("⚠️ WebSocket connection closed:", e)

@app.websocket("/ws/stream-v3")
async def stream_v3(websocket: WebSocket):
    """
    Browser -> (binary 16k PCM) -> FastAPI -> AssemblyAI v3 -> transcripts
    Prints transcripts in server console and forwards JSON to browser.
    """
    await websocket.accept()
    ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY") or os.getenv("ASSEMBLY_API_KEY") or os.getenv("ASSEMBLY_API_KEY")
    # ^ use whichever name you actually set; you had ASSEMBLY_API_KEY yesterday.

    if not ASSEMBLY_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Missing ASSEMBLY_API_KEY"}))
        await websocket.close()
        return

    # AssemblyAI Streaming v3 endpoint (expects 16k, 16-bit, mono PCM in binary frames)
    aai_url = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=true"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(
                aai_url,
                headers={"Authorization": ASSEMBLY_API_KEY},
                heartbeat=20,  # keepalive
            ) as aai_ws:

                print("✅ Connected to AssemblyAI v3 streaming")

                async def forward_client_audio():
                    """Browser -> Server -> AssemblyAI"""
                    try:
                        while True:
                            data = await websocket.receive_bytes()
                            # forward raw PCM (16-bit) bytes directly
                            await aai_ws.send_bytes(data)
                    except Exception as e:
                        # try to terminate AAI session gracefully
                        try:
                            await aai_ws.send_str(json.dumps({"type": "Terminate"}))
                        except:
                            pass
                        await aai_ws.close()
                        # close client socket too
                        await websocket.close()

                async def forward_aai_messages():
                    """AssemblyAI -> Server -> Browser; also print to console"""
                    try:
                        async for msg in aai_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                except:
                                    payload = {"raw": msg.data}

                                mtype = payload.get("type")
                                if mtype == "Begin":
                                    sid = payload.get("id")
                                    exp = payload.get("expires_at")
                                    print(f"\n🎬 Session Begin: id={sid} expires_at={exp}")
                                elif mtype == "Turn":
                                    txt = payload.get("transcript", "")
                                    formatted = payload.get("turn_is_formatted", False)
                                    if formatted:
                                        print(f"\n📝 {txt}")
                                    else:
                                        # live partial
                                        print(f"\r… {txt}", end="", flush=True)
                                elif mtype == "Termination":
                                    adur = payload.get("audio_duration_seconds", 0)
                                    sdur = payload.get("session_duration_seconds", 0)
                                    print(f"\n🏁 Terminated: audio={adur}s session={sdur}s")

                                # forward JSON to browser UI
                                await websocket.send_text(msg.data)

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print("❌ AAI websocket error")
                                break
                    except Exception as e:
                        print(f"❌ Error from AAI stream: {e}")
                        try:
                            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                        except:
                            pass
                        await websocket.close()

                await asyncio.gather(forward_client_audio(), forward_aai_messages())

        except Exception as e:
            print(f"❌ Could not connect to AssemblyAI: {e}")
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": "Failed to connect to AssemblyAI"}))
            finally:
                await websocket.close()