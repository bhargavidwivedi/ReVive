"""
ReVive — Voice Clinical Notes
==============================
Pipeline: Speech → Text (Whisper) → LLM Analysis → Risk Features
Doctors can speak discharge notes directly into the system.
"""

import os
import json
import tempfile
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "voice_notes")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. TRANSCRIBE AUDIO ───────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio file to text using OpenAI Whisper.
    Supports: mp3, mp4, wav, m4a, webm, ogg
    audio_path: path to audio file
    """
    try:
        import whisper
        print("  🎙️  Loading Whisper model (first time may take a minute)...")
        model = whisper.load_model("base")  # base = fast + accurate enough
        print("  🎙️  Transcribing audio...")
        result = model.transcribe(audio_path)
        text   = result["text"].strip()
        print(f"  ✅ Transcribed: {len(text)} characters")
        return text
    except ImportError:
        print("  ⚠️  Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        return None


# ── 2. SIMULATE VOICE INPUT (for testing without microphone) ──────────────────

def simulate_voice_input(text: str) -> str:
    """
    Simulates voice input by returning the text directly.
    Use this for testing without a microphone.
    """
    print(f"  🎙️  [SIMULATED] Voice input received: {len(text)} characters")
    return text


# ── 3. ANALYZE TRANSCRIBED NOTES ─────────────────────────────────────────────

def analyze_transcribed_notes(transcription: str, patient_id: str = "unknown") -> dict:
    """
    Sends transcribed clinical notes to Claude for full analysis.
    Extracts structured risk features + patient-friendly explanation.
    """
    prompt = f"""You are a clinical AI analyzing transcribed doctor's voice notes.

Transcribed clinical notes:
"{transcription}"

Extract ALL clinical information and return ONLY this JSON:
{{
    "structured_data": {{
        "diagnoses": ["diagnosis1", "diagnosis2"],
        "medications": ["med1", "med2"],
        "procedures": ["proc1"],
        "vitals": {{"bp": "value or null", "hr": "value or null", "temp": "value or null"}},
        "allergies": ["allergy1"] or [],
        "follow_up_arranged": true or false,
        "discharge_disposition": "home/SNF/rehab/other"
    }},
    "risk_signals": {{
        "social_isolation_score"      : 0-3,
        "medication_noncompliance_risk": 0-2,
        "mental_health_flag"          : 0 or 1,
        "caregiver_support"           : 0-2,
        "follow_up_arranged"          : 0 or 1,
        "overall_psychosocial_risk"   : 0-10
    }},
    "key_concerns": ["concern1", "concern2", "concern3"],
    "recommended_interventions": ["intervention1", "intervention2"],
    "patient_friendly_summary": "2 sentences explaining the situation in simple English",
    "clinical_summary": "1-2 sentence clinical documentation note",
    "transcription_confidence": "high/medium/low"
}}"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1200,
            messages   = [{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text.strip())
        result["raw_transcription"] = transcription
        result["patient_id"]        = patient_id

        # Save
        path = os.path.join(OUTPUT_DIR, f"voice_note_{patient_id}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✅ Voice note analysis saved → {path}")
        return result

    except Exception as e:
        return {"error": str(e), "raw_transcription": transcription}


# ── 4. FULL VOICE PIPELINE ────────────────────────────────────────────────────

def process_voice_note(
    audio_path  : str = None,
    text_input  : str = None,
    patient_id  : str = "unknown",
) -> dict:
    """
    Full voice note pipeline:
    - If audio_path provided: transcribe then analyze
    - If text_input provided: simulate voice (for testing)
    Returns structured clinical data ready for the agent.
    """
    print(f"\n  🎙️  Processing voice clinical note for patient {patient_id}")

    # Get transcription
    if audio_path and os.path.exists(audio_path):
        transcription = transcribe_audio(audio_path)
    elif text_input:
        transcription = simulate_voice_input(text_input)
    else:
        print("  ❌ No audio file or text input provided")
        return {}

    if not transcription:
        return {}

    print(f"\n  📝 Transcription:\n  \"{transcription[:200]}...\"" if len(transcription) > 200
          else f"\n  📝 Transcription:\n  \"{transcription}\"")

    # Analyze
    print("\n  🧠 Analyzing with Claude...")
    result = analyze_transcribed_notes(transcription, patient_id)

    if "error" not in result:
        print(f"\n  ✅ Voice note processed successfully")
        print(f"  → Diagnoses found: {len(result.get('structured_data', {}).get('diagnoses', []))}")
        print(f"  → Medications found: {len(result.get('structured_data', {}).get('medications', []))}")
        print(f"  → Psychosocial risk: {result.get('risk_signals', {}).get('overall_psychosocial_risk', 0)}/10")
        print(f"  → Patient summary: {result.get('patient_friendly_summary', 'N/A')}")

    return result