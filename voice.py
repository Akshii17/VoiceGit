from __future__ import annotations

import json
import queue
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "vosk-model-small-en-us-0.15"


def listen_and_transcribe() -> str | None:
    """
    Lazy-import sounddevice/vosk here so importing main.py does not init PortAudio.
    """
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except ImportError as e:
        raise RuntimeError(
            "Voice dependencies missing. Install with: pip install -r requirements.txt"
        ) from e

    if not MODEL_DIR.exists():
        print("Vosk model not found. Please download it.")
        return None

    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
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
            # Wait longer (~20s) before declaring no speech.
            for _ in range(40):
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
    except OSError as e:
        raise RuntimeError(
            f"Audio device error (PortAudio): {e}. Try unplugging USB audio or run without voice mode."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Microphone or voice processing error: {e}") from e
