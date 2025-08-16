import os
import requests
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
MURF_API_KEY = os.getenv("MURF_API_KEY")
def generate_tts_with_murf(text, voice_id=None):
    """Generate TTS audio using Murf API"""
    try:
        # Check if API key is available
        if not MURF_API_KEY or MURF_API_KEY.strip() == "":
            raise Exception("MURF_API_KEY not found in environment variables")
        
        # Clean the API key (remove any whitespace)
        api_key = MURF_API_KEY.strip()
        
        # Murf API endpoint
        url = "https://api.murf.ai/v1/speech/generate"
        
        # Use a valid voice ID - default to a common one
        if not voice_id or (isinstance(voice_id, str) and voice_id.strip() == ""):
            voice_id = "en-IN-aarav"
        
        # Clean the voice_id if it's a string
        if isinstance(voice_id, str):
            voice_id = voice_id.strip()
        
        # Try different header formats - Murf API might expect different formats
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try with "voice" field first (as used in other implementations)
        payload = {
            "text": text,
            "voice": voice_id,
            "format": "mp3",
            "sampleRate": 24000
        }
        
        logger.info(f"Generating TTS for text: {text[:50]}...")
        logger.info(f"Using Murf API key: {api_key[:10]}...")
        logger.info(f"Using voice ID: {voice_id}")
        logger.info(f"Payload: {payload}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            # If "voice" field fails, try with "voiceId" field
            logger.info("'voice' field failed, trying with 'voiceId' field...")
            payload = {
                "text": text,
                "voiceId": voice_id,
                "format": "mp3",
                "sampleRate": 24000
            }
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                # If Bearer token fails, try with api-key header
                logger.info("Bearer token failed, trying with api-key header...")
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Murf API error: {response.status_code} - {response.text}")
                    raise Exception(f"Murf API error: {response.status_code} - {response.text}")
        
        result = response.json()
        logger.info(f"Murf API response: {result}")
        
        # Check for different possible response structures
        audio_url = None
        if "audioFile" in result:
            audio_url = result["audioFile"]
        
        if not audio_url:
            logger.error(f"Murf API response missing audioUrl. Full response: {result}")
            raise Exception(f"Murf API response missing audioUrl. Response: {result}")
        
        logger.info(f"TTS generation successful: {audio_url}")
        return audio_url
        
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise
