from __future__ import annotations

import time
from pathlib import Path


STREAM_FILE = Path(__file__).resolve().parent / "learning_stream.txt"


def main() -> None:
    print("Learning window started. Listening for explanations...")
    STREAM_FILE.touch(exist_ok=True)

    with STREAM_FILE.open("r", encoding="utf-8") as f:
        # Show only new lines from now on.
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                print(line.rstrip())
            else:
                time.sleep(0.4)


if __name__ == "__main__":
    main()

