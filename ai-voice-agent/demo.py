import traceback
import os
import tempfile
import time
import requests
import json
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List

from services.tts_murf import generate_tts_with_murf # type: ignore
from services.tts_murf_ws import stream_tts_with_murf_ws, get_active_session_count, cleanup_all_sessions
from services.spiderman import SPIDERMAN_PROMPT

from fastapi import WebSocket, WebSocketDisconnect
import uuid
from datetime import datetime

import aiohttp
import asyncio

# Load ENV & Configure APIs
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
# MURF_API_KEY = os.getenv("MURF_API_KEY")

# if not GEMINI_API_KEY or not ASSEMBLY_API_KEY or not MURF_API_KEY:
#     raise RuntimeError("❗ Missing one or more API keys in environment variables")

# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# DAY 27 Dynamic Api keys
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
DEFAULT_MURF_API_KEY = os.getenv("MURF_API_KEY")

if not DEFAULT_GEMINI_API_KEY or not DEFAULT_ASSEMBLY_API_KEY or not DEFAULT_MURF_API_KEY:
    raise RuntimeError("❗ Missing one or more API keys in environment variables")

genai.configure(api_key=DEFAULT_GEMINI_API_KEY)
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
# DAY 27 Dynamic API Keys
app.state.GEMINI_API_KEY = DEFAULT_GEMINI_API_KEY
app.state.ASSEMBLY_API_KEY = DEFAULT_ASSEMBLY_API_KEY
app.state.MURF_API_KEY = DEFAULT_MURF_API_KEY
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Day 27 Dynamic API Key allocation
@app.post("/configure-api-keys")
async def configure_api_keys(request: Request):
    data = await request.json()
    gemini = data.get("gemini")
    assembly = data.get("assembly")
    murf = data.get("murf")

    # Update only the keys that are present
    if gemini:
        app.state.GEMINI_API_KEY = gemini
        genai.configure(api_key=gemini)  # reconfigure Gemini immediately
        print(f"GEMINI_API_KEY configured ✅\n{gemini}")

    if assembly:
        app.state.ASSEMBLY_API_KEY = assembly
        print(f"ASSEMBLY_API_KEY configured ✅")

    if murf:
        app.state.MURF_API_KEY = murf
        print(f"MURF_API_KEY configured ✅")

    if not (gemini or assembly or murf):
        raise HTTPException(status_code=400, detail="At least one API key must be provided")

    # for all three api update
    # if not gemini or not assembly or not murf:
    #     raise HTTPException(status_code=400, detail="All API keys are required")

    # # Store in app state
    # app.state.GEMINI_API_KEY = gemini
    # app.state.ASSEMBLY_API_KEY = assembly
    # app.state.MURF_API_KEY = murf

    # Reconfigure Gemini dynamically
    # genai.configure(api_key=gemini)

    return {"message": "API keys updated successfully"}

# Day 25
async def get_superhero_fun_fact(hero_name: str, websocket: WebSocket):
    """
    Finds and streams a fun fact about a specific superhero using Gemini with Google Search.
    """
    try:
        # Define the system prompt for the fun fact skill.
        system_prompt = f"""
        You are a friendly and knowledgeable AI assistant. Your task is to provide a single, concise, and fun fact about the superhero "{hero_name}". The fact should be surprising or interesting. Do not include any personal commentary or conversational phrases. Just state the fact.
        """
        
        # Define the user query for the search.
        user_query = f"Provide a fun fact about the superhero {hero_name}."
        
        # Initialize the Gemini model without the tools argument.
        # The tools parameter is passed directly to the generate_content call.
        gemini_grounded_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        # Make the API call with the system prompt and user query.
        # The `tools` parameter is now correctly included here.
        response = await asyncio.to_thread(
            lambda: gemini_grounded_model.generate_content(
                # The system instruction is now part of the user's content.
                f"{system_prompt}\n{user_query}",
                # tools=[{"google_search": {}}],  # <-- The tools parameter is still here.
                stream=False
            )
        )
        
        fact_text = response.candidates[0].content.parts[0].text
        
        # Send the fun fact as a single block of text to the client.
        await websocket.send_text(json.dumps({
            "type": "llm_delta",
            "text": fact_text
        }))

        # Signal LLM completion for this skill.
        await websocket.send_text(json.dumps({
            "type": "llm_done",
            "full_text": fact_text
        }))
        
        # Convert the fun fact to speech.
        # This part of the code assumes you have a working `stream_tts_with_murf_ws` function.
        await stream_tts_with_murf_ws(
            text=fact_text,
            websocket=websocket,
            voice_id="en-US-miles",  # or your preferred voice
            sample_rate=44100,
            max_retries=2,
            use_unique_context=True
        )
        
        print(f"✅ Fun fact provided for {hero_name}.")

    except Exception as e:
        print(f"❌ Error getting fun fact: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Sorry, I couldn't find a fun fact right now. My web-shooters must be jammed!"
        }))

# Day 26
async def google_search(search_query: str, websocket: WebSocket):
    """
    Finds and streams a fun fact about a specific superhero using Gemini with Google Search.
    """
    print("hello from google search")
    try:
        # Define the system prompt for the fun fact skill.
        system_prompt = f"""You are a helpful assistant. Your job:
            1. Use the Google Search tool when needed.
            2. Provide a single, concise, interesting fact about "{search_query}".
            3. No jokes, no placeholders, no extra commentary.
            Return only the fact.
            """
        
        # Define the user query for the search.
        user_query = f"Use Google search to find the most recent and relevant information about: '{search_query}'. Summarize the findings."
        
        # Initialize the Gemini model without the tools argument.
        # The tools parameter is passed directly to the generate_content call.
        gemini_grounded_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        # Make the API call with the system prompt and user query.
        # The `tools` parameter is now correctly included here.
        response = await asyncio.to_thread(
            lambda: gemini_grounded_model.generate_content(
                # The system instruction is now part of the user's content.
                f"{system_prompt}\n{user_query}",
                # tools=[{"google_search": {}}],  # <-- The tools parameter is still here.
                stream=False
            )
        )

        
        fact_text = response.candidates[0].content.parts[0].text
        
        # Send the fun fact as a single block of text to the client.
        await websocket.send_text(json.dumps({
            "type": "llm_delta",
            "text": fact_text
        }))

        # Signal LLM completion for this skill.
        await websocket.send_text(json.dumps({
            "type": "llm_done",
            "full_text": fact_text
        }))
        
        # Convert the fun fact to speech.
        # This part of the code assumes you have a working `stream_tts_with_murf_ws` function.
        await stream_tts_with_murf_ws(
            text=fact_text,
            websocket=websocket,
            voice_id="en-US-miles",  # or your preferred voice
            sample_rate=44100,
            max_retries=2,
            use_unique_context=True
        )
        
        print(f"✅ Result provided for {search_query} from GOOGLE: ")

    except Exception as e:
        print(f"❌ Error getting fun fact: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Sorry, I couldn't find a fun fact right now. My web-shooters must be jammed!"
        }))


# --- NEW: stream Gemini tokens helper ---
async def stream_gemini_response(text: str, websocket: WebSocket):
    """
    Stream Gemini response and then convert to speech with improved error handling
    """
    try:
        # --- NEW SKILL-BASED LOGIC ---
        # Check if the user is asking for a "fun fact about" a superhero. # Day 25
        if "fun fact about" in text.lower():
            print("not hello")
            # Extract the superhero's name from the user's message.
            hero_name = text.lower().split("fun fact about", 1)[1].strip().replace("?", "").replace("!", "")
            if hero_name:
                print(f"🔍 User requested a fun fact about: {hero_name}")
                await get_superhero_fun_fact(hero_name, websocket)
                return  # Exit the function after handling the skill.
            
        # Day 26 # Example: Detect if user wants a Google search
        if "search for" in text.lower() or "google" in text.lower():
            print("🔍 Google Search requested")
            
            # Extract search query after the keyword
            search_query = text.lower().split("search for", 1)[1].strip().replace("?", "").replace("!", "")
            
            if search_query:
                print(f"🔍 Performing Google Search for: {search_query}")
                await google_search(search_query, websocket)
                return  # Exit after handling

        # Check TTS session limit before starting LLM
        active_sessions = await get_active_session_count()
        if active_sessions >= 2:
            print(f"⚠️ High TTS load ({active_sessions} active), will use fallback if needed")
        
        # Your existing Gemini streaming code here
        # For example:
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # model = genai.GenerativeModel('gemini-pro')

        # Spiderman persona
        persona_prompt=f"{SPIDERMAN_PROMPT}\n\nUser: {text}\nSpiderman:"
        
        # Stream the response
        response = gemini_model.generate_content(persona_prompt, stream=True)
        full_response = ""
        
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                # Send streaming text to client
                await websocket.send_text(json.dumps({
                    "type": "llm_delta",
                    "text": chunk.text
                }))
        
        # Signal LLM completion
        await websocket.send_text(json.dumps({
            "type": "llm_done",
            "full_text": full_response
        }))
        
        print(f"🤖 Generated response ({len(full_response)} chars): {full_response[:100]}...")
        
        # Convert to speech with improved error handling
        print("🎤 Starting TTS conversion...")
        await stream_tts_with_murf_ws(
            text=full_response,
            websocket=websocket,
            voice_id="en-US-miles",  # or your preferred voice
            sample_rate=44100,
            max_retries=2,
            use_unique_context=True  # Use unique context to avoid limit issues
        )
        
    except Exception as e:
        print(f"❌ Error in Gemini response: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "AI processing failed"
        }))
        raise

@app.on_event("startup")
async def startup_event():
    print("🚀 Application starting up...")
    # Your other startup code here

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Application shutting down...")
    await cleanup_all_sessions()

@app.get("/", response_class=HTMLResponse)
async def root():
    # return FileResponse(static_path / "audio.html") # for day 16 receive audio chunks
    return FileResponse(static_path / "spidey.html") # for day 24
    return FileResponse(static_path / "serve.html") # for day 22 or 23
    # return FileResponse(static_path / "transcribe.html") # for day 17
    # return FileResponse(static_path / "index.html") # default file response

FALLBACK_MESSAGE="I'm having trouble connecting right now."

# Original WebSocket endpoint (kept for compatibility)
@app.websocket("/ws/stream-v3")
async def stream_v3(websocket: WebSocket):
    """
    Improved continuous listening WebSocket with better TTS session management.
    """
    await websocket.accept()
    print("🔗 Continuous listening WebSocket established")
    
    # ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
    ASSEMBLY_API_KEY = getattr(app.state, "ASSEMBLY_API_KEY") # DAY 27 - Dynamic Keys
    if not ASSEMBLY_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Missing ASSEMBLY_API_KEY"}))
        await websocket.close()
        return

    # Global session state
    client_active = True
    conversation_count = 0
    
    async def is_client_active():
        """Check if client is still connected"""
        try:
            return client_active and websocket.application_state.value == 1
        except:
            return False
    
    async def safe_send_to_client(data):
        """Safely send data to client"""
        try:
            if await is_client_active():
                if isinstance(data, dict):
                    await websocket.send_text(json.dumps(data))
                else:
                    await websocket.send_text(data)
                return True
        except Exception as e:
            print(f"❌ Client send failed: {e}")
            nonlocal client_active
            client_active = False
        return False

    # Main continuous listening loop
    while await is_client_active():
        conversation_count += 1
        
        # Check TTS session load before starting conversation
        active_tts_sessions = await get_active_session_count()
        if active_tts_sessions > 0:
            print(f"ℹ️ Starting conversation #{conversation_count} (TTS sessions active: {active_tts_sessions})")
        else:
            print(f"\n🎬 Starting conversation #{conversation_count}")
        
        # Create new AAI session for each conversation
        session = None
        aai_ws = None
        
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30),
                connector=aiohttp.TCPConnector(keepalive_timeout=30)
            )
            
            # Connect to AssemblyAI for this conversation
            aai_url = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=true"
            aai_ws = await session.ws_connect(
                aai_url,
                headers={"Authorization": app.state.ASSEMBLY_API_KEY},
                heartbeat=30,
                timeout=30
            )
            print(f"✅ AAI connected for conversation #{conversation_count}")
            
            # Send ready signal to client
            await safe_send_to_client({
                "type": "ready",
                "conversation": conversation_count,
                "message": f"Ready for conversation #{conversation_count}",
                "tts_sessions_active": active_tts_sessions
            })
            
            # Track this conversation's state
            conversation_active = True
            audio_forwarding_active = True
            final_transcript_received = False
            llm_processing_complete = False
            tts_processing = False
            
            async def forward_audio_to_aai():
                """Forward client audio to AssemblyAI for this conversation"""
                nonlocal audio_forwarding_active, conversation_active
                
                print(f"🎤 Starting audio forwarding for conversation #{conversation_count}")
                
                while conversation_active and await is_client_active():
                    try:
                        # Check AAI connection health
                        if aai_ws.closed:
                            print("⚠️ AAI connection closed, stopping audio forwarding")
                            break
                        
                        # Adaptive timeout based on TTS processing
                        timeout = 0.2 if tts_processing else 1.0
                        
                        data = await asyncio.wait_for(
                            websocket.receive_bytes(), 
                            timeout=timeout
                        )
                        
                        # Only forward if conversation is still active and AAI is connected
                        if conversation_active and not aai_ws.closed:
                            await aai_ws.send_bytes(data)
                        else:
                            break
                            
                    except asyncio.TimeoutError:
                        if not conversation_active:
                            break
                        continue
                        
                    except WebSocketDisconnect:
                        print("🔌 Client disconnected during audio forwarding")
                        nonlocal client_active
                        client_active = False
                        conversation_active = False
                        break
                        
                    except Exception as e:
                        print(f"⚠️ Audio forwarding error: {e}")
                        await asyncio.sleep(0.1)
                        continue
                
                audio_forwarding_active = False
                print(f"🔇 Audio forwarding stopped for conversation #{conversation_count}")

            async def process_aai_messages():
                """Process AssemblyAI messages for this conversation"""
                nonlocal final_transcript_received, llm_processing_complete, conversation_active, tts_processing
                
                try:
                    async for msg in aai_ws:
                        if not await is_client_active():
                            break
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                            except:
                                continue
                            
                            mtype = payload.get("type")
                            
                            if mtype == "Begin":
                                sid = payload.get("id", "unknown")
                                print(f"🎬 AAI Session Begin: {sid}")
                                
                            elif mtype == "Turn":
                                txt = payload.get("transcript", "").strip()
                                formatted = payload.get("turn_is_formatted", False)
                                
                                if formatted and txt:
                                    print(f"📝 Final Turn: {txt}")
                                    final_transcript_received = True
                                    
                                    # Send transcript to client
                                    await safe_send_to_client({
                                        "type": "turn_end",
                                        "final_transcript": txt,
                                        "conversation": conversation_count
                                    })
                                    
                                    # Start TTS processing flag
                                    tts_processing = True
                                    
                                    # Process with LLM and TTS
                                    print("🤖 Processing with Gemini...")
                                    try:
                                        await stream_gemini_response(txt, websocket)
                                        llm_processing_complete = True
                                        tts_processing = False
                                        print("✅ LLM and TTS processing complete")
                                        
                                        # Give a moment for audio to start playing
                                        await asyncio.sleep(2)
                                        
                                        # Signal that this conversation is done
                                        await safe_send_to_client({
                                            "type": "conversation_complete",
                                            "conversation": conversation_count,
                                            "message": "Ready for next question"
                                        })
                                        
                                        # End this conversation after successful processing
                                        conversation_active = False
                                        print(f"🏁 Conversation #{conversation_count} marked complete")
                                        return
                                        
                                    except Exception as llm_error:
                                        print(f"❌ LLM/TTS error: {llm_error}")
                                        tts_processing = False
                                        await safe_send_to_client({
                                            "type": "error",
                                            "message": "AI processing failed - please try again"
                                        })
                                        llm_processing_complete = True
                                        conversation_active = False
                                        return
                                
                                else:
                                    # Live partial transcript
                                    if txt:
                                        print(f"\r… {txt}", end="", flush=True)
                                        await safe_send_to_client({
                                            "type": "partial",
                                            "text": txt
                                        })
                                
                                # Always forward raw turn data
                                await safe_send_to_client(msg.data)
                                
                            elif mtype == "Termination":
                                print(f"\n🏁 AAI Session terminated for conversation #{conversation_count}")
                                await safe_send_to_client(msg.data)
                                conversation_active = False
                                return
                                
                            else:
                                # Forward other AAI messages
                                await safe_send_to_client(msg.data)
                        
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                            print(f"❌ AAI connection issue for conversation #{conversation_count}")
                            conversation_active = False
                            return
                            
                except Exception as e:
                    print(f"❌ AAI message processing error: {e}")
                    conversation_active = False

            # Run both audio forwarding and message processing concurrently
            print(f"🚀 Starting conversation #{conversation_count} tasks")
            
            try:
                results = await asyncio.gather(
                    forward_audio_to_aai(),
                    process_aai_messages(),
                    return_exceptions=True
                )
                
                # Log task results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        task_name = ["audio_forwarding", "message_processing"][i]
                        print(f"⚠️ Task {task_name} failed: {result}")
                        
            except Exception as e:
                print(f"❌ Conversation #{conversation_count} task error: {e}")
            
            # Ensure conversation is marked as inactive
            conversation_active = False
            
            print(f"✅ Conversation #{conversation_count} tasks completed")
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to create AAI session for conversation #{conversation_count}: {e}")
            await safe_send_to_client({
                "type": "error", 
                "message": f"Failed to start conversation: {str(e)}"
            })
            await asyncio.sleep(3)
            
        finally:
            # Cleanup AAI WebSocket
            if aai_ws and not aai_ws.closed:
                try:
                    await aai_ws.close()
                except:
                    pass
            
            # Cleanup HTTP session
            if session and not session.closed:
                try:
                    await session.close()
                except:
                    pass

    # Final cleanup
    print(f"🏁 Continuous listening ended after {conversation_count} conversations")
    await cleanup_all_sessions()  # Clean up any remaining TTS sessions
    
    try:
        await websocket.send_text(json.dumps({
            "type": "session_ended",
            "total_conversations": conversation_count
        }))
    except:
        pass
    """
    Truly continuous listening WebSocket that stays active for multiple queries.
    Fixed connection management and proper cleanup.
    """
    await websocket.accept()
    print("🔗 Continuous listening WebSocket established")
    
    # ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
    ASSEMBLY_API_KEY = getattr(app.state, "ASSEMBLY_API_KEY") # DAY 27 - Dynamic API Keys
    if not ASSEMBLY_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Missing ASSEMBLY_API_KEY"}))
        await websocket.close()
        return

    # Global session state
    client_active = True
    conversation_count = 0
    
    async def is_client_active():
        """Check if client is still connected"""
        try:
            return client_active and websocket.application_state.value == 1
        except:
            return False
    
    async def safe_send_to_client(data):
        """Safely send data to client"""
        try:
            if await is_client_active():
                if isinstance(data, dict):
                    await websocket.send_text(json.dumps(data))
                else:
                    await websocket.send_text(data)
                return True
        except Exception as e:
            print(f"❌ Client send failed: {e}")
            nonlocal client_active
            client_active = False
        return False

    # Main continuous listening loop
    while await is_client_active():
        conversation_count += 1
        print(f"\n🎬 Starting conversation #{conversation_count}")
        
        # Create new AAI session for each conversation
        session = None
        aai_ws = None
        
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30),
                connector=aiohttp.TCPConnector(keepalive_timeout=30)
            )
            
            # Connect to AssemblyAI for this conversation
            aai_url = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=true"
            aai_ws = await session.ws_connect(
                aai_url,
                headers={"Authorization": ASSEMBLY_API_KEY},
                heartbeat=30,
                timeout=30
            )
            print(f"✅ AAI connected for conversation #{conversation_count}")
            
            # Send ready signal to client
            await safe_send_to_client({
                "type": "ready",
                "conversation": conversation_count,
                "message": f"Ready for conversation #{conversation_count}"
            })
            
            # Track this conversation's state
            conversation_active = True
            audio_forwarding_active = True
            final_transcript_received = False
            llm_processing_complete = False
            tts_processing = False
            
            async def forward_audio_to_aai():
                """Forward client audio to AssemblyAI for this conversation"""
                nonlocal audio_forwarding_active, conversation_active
                
                print(f"🎤 Starting audio forwarding for conversation #{conversation_count}")
                
                while conversation_active and await is_client_active():
                    try:
                        # Check AAI connection health
                        if aai_ws.closed:
                            print("⚠️ AAI connection closed, stopping audio forwarding")
                            break
                        
                        # Non-blocking receive with shorter timeout during TTS
                        timeout = 0.2 if tts_processing else 1.0
                        
                        data = await asyncio.wait_for(
                            websocket.receive_bytes(), 
                            timeout=timeout
                        )
                        
                        # Only forward if conversation is still active and AAI is connected
                        if conversation_active and not aai_ws.closed:
                            await aai_ws.send_bytes(data)
                        else:
                            print("⚠️ Skipping audio forward - conversation ending or AAI closed")
                            break
                            
                    except asyncio.TimeoutError:
                        # No audio data - continue listening unless conversation is done
                        if not conversation_active:
                            break
                        continue
                        
                    except WebSocketDisconnect:
                        print("🔌 Client disconnected during audio forwarding")
                        nonlocal client_active
                        client_active = False
                        conversation_active = False
                        break
                        
                    except ConnectionResetError:
                        print("🔌 Connection reset during audio forwarding")
                        conversation_active = False
                        break
                        
                    except Exception as e:
                        print(f"⚠️ Audio forwarding error: {e}")
                        # Don't break on minor errors, just continue
                        await asyncio.sleep(0.1)
                        continue
                
                audio_forwarding_active = False
                print(f"🔇 Audio forwarding stopped for conversation #{conversation_count}")

            async def process_aai_messages():
                """Process AssemblyAI messages for this conversation"""
                nonlocal final_transcript_received, llm_processing_complete, conversation_active, tts_processing
                
                try:
                    async for msg in aai_ws:
                        if not await is_client_active():
                            break
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                            except:
                                continue
                            
                            mtype = payload.get("type")
                            
                            if mtype == "Begin":
                                sid = payload.get("id", "unknown")
                                print(f"🎬 AAI Session Begin: {sid}")
                                
                            elif mtype == "Turn":
                                txt = payload.get("transcript", "").strip()
                                formatted = payload.get("turn_is_formatted", False)
                                
                                if formatted and txt:
                                    print(f"📝 Final Turn: {txt}")
                                    final_transcript_received = True
                                    
                                    # Send transcript to client
                                    await safe_send_to_client({
                                        "type": "turn_end",
                                        "final_transcript": txt,
                                        "conversation": conversation_count
                                    })
                                    
                                    # Start TTS processing flag
                                    tts_processing = True
                                    
                                    # Process with LLM and TTS
                                    print("🤖 Processing with Gemini...")
                                    try:
                                        await stream_gemini_response(txt, websocket)
                                        llm_processing_complete = True
                                        tts_processing = False
                                        print("✅ LLM and TTS processing complete")
                                        
                                        # Give a moment for audio to start playing
                                        await asyncio.sleep(2)
                                        
                                        # Signal that this conversation is done
                                        await safe_send_to_client({
                                            "type": "conversation_complete",
                                            "conversation": conversation_count,
                                            "message": "Ready for next question"
                                        })
                                        
                                        # End this conversation after successful processing
                                        conversation_active = False
                                        print(f"🏁 Conversation #{conversation_count} marked complete")
                                        return
                                        
                                    except Exception as llm_error:
                                        print(f"❌ LLM error: {llm_error}")
                                        tts_processing = False
                                        await safe_send_to_client({
                                            "type": "error",
                                            "message": "LLM processing failed"
                                        })
                                        llm_processing_complete = True
                                        conversation_active = False
                                        return
                                
                                else:
                                    # Live partial transcript
                                    if txt:
                                        print(f"\r… {txt}", end="", flush=True)
                                        await safe_send_to_client({
                                            "type": "partial",
                                            "text": txt
                                        })
                                
                                # Always forward raw turn data
                                await safe_send_to_client(msg.data)
                                
                            elif mtype == "Termination":
                                print(f"\n🏁 AAI Session terminated for conversation #{conversation_count}")
                                await safe_send_to_client(msg.data)
                                conversation_active = False
                                return
                                
                            else:
                                # Forward other AAI messages
                                await safe_send_to_client(msg.data)
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"❌ AAI WebSocket error for conversation #{conversation_count}")
                            conversation_active = False
                            return
                        
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            print(f"🔌 AAI WebSocket closed for conversation #{conversation_count}")
                            conversation_active = False
                            return
                            
                except Exception as e:
                    print(f"❌ AAI message processing error: {e}")
                    conversation_active = False

            # Run both audio forwarding and message processing concurrently
            print(f"🚀 Starting conversation #{conversation_count} tasks")
            
            try:
                # Use gather with return_exceptions to handle task failures gracefully
                results = await asyncio.gather(
                    forward_audio_to_aai(),
                    process_aai_messages(),
                    return_exceptions=True
                )
                
                # Check for exceptions in results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        task_name = ["audio_forwarding", "message_processing"][i]
                        print(f"⚠️ Task {task_name} failed: {result}")
                        
            except Exception as e:
                print(f"❌ Conversation #{conversation_count} task error: {e}")
            
            # Ensure conversation is marked as inactive
            conversation_active = False
            
            # Wait for cleanup
            print(f"✅ Conversation #{conversation_count} tasks completed")
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to create AAI session for conversation #{conversation_count}: {e}")
            await safe_send_to_client({
                "type": "error",
                "message": f"Failed to start conversation #{conversation_count}: {str(e)}"
            })
            await asyncio.sleep(3)  # Wait before retry
            
        finally:
            # Cleanup AAI WebSocket
            if aai_ws and not aai_ws.closed:
                try:
                    await aai_ws.close()
                    print(f"🧹 AAI WebSocket closed for conversation #{conversation_count}")
                except:
                    pass
            
            # Cleanup HTTP session
            if session and not session.closed:
                try:
                    await session.close()
                    print(f"🧹 HTTP session closed for conversation #{conversation_count}")
                except:
                    pass
            
            print(f"🧹 Cleanup completed for conversation #{conversation_count}")

    # Final cleanup
    print(f"🏁 Continuous listening ended after {conversation_count} conversations")
    try:
        await websocket.send_text(json.dumps({
            "type": "session_ended",
            "total_conversations": conversation_count
        }))
    except:
        pass