from __future__ import annotations

import json
import queue
from pathlib import Path

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from vosk import KaldiRecognizer, Model
except Exception:
    KaldiRecognizer = None
    Model = None


MODEL_DIR = Path(__file__).resolve().parent / "vosk-model-small-en-us-0.15"


def listen_and_transcribe() -> str | None:
    if sd is None or Model is None or KaldiRecognizer is None:
        raise RuntimeError(
            "Voice dependencies missing. Install with: pip install -r requirements.txt"
        )

    if not MODEL_DIR.exists():
        print("Vosk model not found. Please download it.")
        return None

    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            # Keep going; recognizer can still continue.
            print(f"Audio status: {status}")
        q.put(bytes(indata))

    try:
        model = Model(str(MODEL_DIR))
        rec = KaldiRecognizer(model, 16000)

        print("Listening... speak now.")
        heard_any_audio = False
        final_text = ""

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            # Try for up to ~10 seconds.
            for _ in range(20):
                data = q.get(timeout=0.5)
                if data:
                    heard_any_audio = True
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    candidate = (result.get("text") or "").strip()
                    if candidate:
                        final_text = candidate
                        break

        if not final_text:
            partial = json.loads(rec.FinalResult()).get("text", "").strip()
            final_text = partial

        if not heard_any_audio or not final_text:
            print("No speech detected.")
            return None

        print(f"You said: {final_text}")
        while True:
            answer = input("Confirm? (y/n) ").strip().lower()
            if answer in {"y", "yes"}:
                return final_text
            if answer in {"n", "no"}:
                return None
            print("Please enter 'y' or 'n'.")
    except queue.Empty:
        print("No speech detected.")
        return None
    except Exception as e:
        # Let caller fall back to text input.
        raise RuntimeError(f"Microphone or voice processing error: {e}") from e

