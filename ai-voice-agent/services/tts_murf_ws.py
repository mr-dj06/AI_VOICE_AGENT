# services/tts_murf_ws.py (fixed version with session management and fallback)
import os
import json
import aiohttp
import asyncio
import hashlib
import time
from typing import Optional

MURF_API_KEY = os.getenv("MURF_API_KEY")
WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

# Global session tracking
_active_sessions = set()
_session_lock = asyncio.Lock()

async def cleanup_session(session_id: str):
    """Remove session from active tracking"""
    async with _session_lock:
        _active_sessions.discard(session_id)
        print(f"🧹 Cleaned up session: {session_id}")

async def generate_session_id(text: str) -> str:
    """Generate unique session ID"""
    timestamp = str(int(time.time() * 1000))
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"session_{timestamp}_{text_hash}"

async def stream_tts_with_browser_fallback(
    text: str,
    websocket=None
) -> None:
    """
    Fallback TTS using browser's built-in speech synthesis.
    This runs on the client side when Murf fails.
    """
    if not websocket:
        print("⚠️ No websocket for fallback TTS")
        return
    
    try:
        print("🗣️ Using browser fallback TTS")
        await websocket.send_text(json.dumps({
            "type": "tts_fallback",
            "text": text,
            "message": "Using browser speech synthesis"
        }))
        print("✅ Fallback TTS initiated on client")
    except Exception as e:
        print(f"❌ Fallback TTS failed: {e}")

async def stream_tts_with_murf_ws(
    text: str,
    websocket=None,
    *,
    voice_id: str = "en-US-amara",
    sample_rate: int = 44100,
    channel_type: str = "MONO",
    audio_format: str = "WAV",
    max_retries: int = 2,
    use_unique_context: bool = True
) -> None:
    """
    Send text to Murf via WebSocket with session management and fallback.
    """
    if not MURF_API_KEY:
        print("❗ MURF_API_KEY missing - using fallback")
        await stream_tts_with_browser_fallback(text, websocket)
        return

    if not text.strip():
        print("⚠️ Empty text provided to TTS")
        return

    # Check active sessions limit
    async with _session_lock:
        if len(_active_sessions) >= 2:  # Murf typically allows 2-3 concurrent sessions
            print(f"⚠️ Too many active sessions ({len(_active_sessions)}), using fallback")
            await stream_tts_with_browser_fallback(text, websocket)
            return

    # Generate unique context/session ID
    session_id = await generate_session_id(text)
    context_id = f"ctx_{session_id}" if use_unique_context else "day22-persistent-context"
    
    # Add to active sessions
    async with _session_lock:
        _active_sessions.add(session_id)
        print(f"🎯 Starting TTS session: {session_id} (active: {len(_active_sessions)})")

    qs = f"?api-key={MURF_API_KEY}&sample_rate={sample_rate}&channel_type={channel_type}&format={audio_format}"
    url = f"{WS_URL}{qs}"

    retry_count = 0
    
    # Check if client is still connected before starting
    client_connected = True
    if websocket:
        try:
            client_connected = websocket.application_state.value == 1
        except:
            client_connected = False
    
    if not client_connected:
        print("⚠️ Client disconnected before TTS started")
        await cleanup_session(session_id)
        return
    
    session = None
    murf_ws = None
    
    try:
        while retry_count < max_retries:
            try:
                print(f"🎤 Connecting to Murf TTS (attempt {retry_count + 1}/{max_retries}) - Session: {session_id}")
                
                # Create session with shorter timeouts to fail faster
                connector = aiohttp.TCPConnector(
                    keepalive_timeout=20,
                    enable_cleanup_closed=True,
                    limit=10,  # Limit concurrent connections
                    limit_per_host=3
                )
                
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=30,  # Reduced total timeout
                        connect=5   # Quick connection timeout
                    ),
                    connector=connector
                )
                
                # Connect to Murf WebSocket
                murf_ws = await session.ws_connect(
                    url,
                    heartbeat=15,
                    timeout=5,
                    max_msg_size=1024*1024*10  # 10MB max message
                )
                print(f"✅ Connected to Murf TTS WebSocket - Session: {session_id}")
                
                # Send context ID
                context_msg = {"context_id": context_id}
                await murf_ws.send_str(json.dumps(context_msg))
                print(f"📋 Context sent: {context_id}")
                
                # Voice configuration
                voice_config_msg = {
                    "voice_config": {
                        "voiceId": voice_id,
                        "style": "Conversational",
                        "rate": 0,
                        "pitch": 0,
                        "variation": 1
                    }
                }
                await murf_ws.send_str(json.dumps(voice_config_msg))
                print("🔧 Voice configuration sent")

                # Send the text
                text_msg = {
                    "text": text,
                    "end": True
                }
                await murf_ws.send_str(json.dumps(text_msg))
                print(f"📝 Text sent: '{text[:50]}{'...' if len(text) > 50 else ''}'")

                chunk_count = 0
                total_audio_size = 0
                audio_started = False
                success = False
                
                # Process incoming audio chunks with timeout
                try:
                    async with asyncio.timeout(25):  # Overall timeout for TTS processing
                        async for msg in murf_ws:
                            # Periodic client connection check
                            if chunk_count % 3 == 0 and websocket:
                                try:
                                    client_connected = websocket.application_state.value == 1
                                    if not client_connected:
                                        print("⚠️ Client disconnected during TTS")
                                        break
                                except:
                                    client_connected = False
                                    break
                            
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                except Exception:
                                    continue

                                if "audio" in data:
                                    audio_b64 = data["audio"]
                                    chunk_count += 1
                                    total_audio_size += len(audio_b64)
                                    
                                    if not audio_started:
                                        print("🎵 First audio chunk received")
                                        audio_started = True
                                    
                                    # Forward to client if still connected
                                    if websocket and client_connected:
                                        try:
                                            await websocket.send_text(json.dumps({
                                                "type": "audio_chunk",
                                                "chunk": audio_b64,
                                                "chunk_number": chunk_count,
                                                "chunk_size": len(audio_b64)
                                            }))
                                        except Exception as send_error:
                                            print(f"❌ Failed to send chunk {chunk_count}: {send_error}")
                                            client_connected = False

                                elif "error" in data:
                                    error_msg = data["error"]
                                    if "context limit" in error_msg.lower() or "active context" in error_msg.lower():
                                        print(f"🚫 Murf context limit hit: {error_msg}")
                                        # Don't retry for context limit errors, use fallback immediately
                                        await stream_tts_with_browser_fallback(text, websocket)
                                        return
                                    else:
                                        raise Exception(f"Murf TTS error: {error_msg}")

                                elif data.get("final"):
                                    print(f"✅ Murf TTS completed! Chunks: {chunk_count}, Size: {total_audio_size}")
                                    success = True
                                    
                                    # Send completion signal
                                    if websocket and client_connected:
                                        try:
                                            await websocket.send_text(json.dumps({
                                                "type": "audio_complete",
                                                "total_chunks": chunk_count,
                                                "total_size": total_audio_size
                                            }))
                                        except:
                                            pass
                                    break

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                raise Exception("Murf WebSocket connection error")
                            elif msg.type == aiohttp.WSMsgType.CLOSE:
                                if chunk_count == 0:
                                    raise Exception("Murf WebSocket closed without audio")
                                break
                
                except asyncio.TimeoutError:
                    raise Exception("Murf TTS processing timeout")
                
                if success:
                    return  # Success - exit function
                else:
                    raise Exception("TTS processing incomplete")

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                # Check for context limit errors
                if "context limit" in error_msg.lower() or "active context" in error_msg.lower():
                    print(f"🚫 Context limit error: {error_msg}")
                    break  # Don't retry context limit errors
                
                print(f"❌ Murf TTS error (attempt {retry_count}/{max_retries}): {error_msg}")
                
                if retry_count < max_retries:
                    delay = min(2 ** retry_count, 5)  # Max 5 second delay
                    print(f"⏳ Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
            
            finally:
                # Cleanup for this attempt
                if murf_ws and not murf_ws.closed:
                    try:
                        await murf_ws.close()
                    except:
                        pass
                
                if session and not session.closed:
                    try:
                        await session.close()
                    except:
                        pass
                
                # Small delay between attempts
                if retry_count < max_retries:
                    await asyncio.sleep(1)

    finally:
        # Always cleanup session tracking
        await cleanup_session(session_id)

    # If we get here, Murf failed - use fallback
    print("🔄 Murf TTS failed, switching to browser fallback")
    await stream_tts_with_browser_fallback(text, websocket)

# Utility function to check active sessions
async def get_active_session_count() -> int:
    """Get current number of active TTS sessions"""
    async with _session_lock:
        return len(_active_sessions)

# Cleanup function for application shutdown
async def cleanup_all_sessions():
    """Clean up all active sessions on shutdown"""
    async with _session_lock:
        session_count = len(_active_sessions)
        _active_sessions.clear()
        if session_count > 0:
            print(f"🧹 Cleaned up {session_count} active TTS sessions on shutdown")