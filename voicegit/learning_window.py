from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText


STREAM_FILE = Path(__file__).resolve().parent / "learning_stream.txt"


def main() -> None:
    STREAM_FILE.touch(exist_ok=True)

    root = tk.Tk()
    root.title("VoiceGit Learning Window")
    root.geometry("700x420")

    # Keep this window visible above other windows.
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()

    text = ScrolledText(root, wrap="word", state="disabled")
    text.pack(fill="both", expand=True, padx=8, pady=8)

    text.configure(state="normal")
    text.insert("end", "Run commands and get explanations...\n")
    text.configure(state="disabled")

    stream = STREAM_FILE.open("r", encoding="utf-8")
    stream.seek(0, 2)  # tail new content only

    def poll_stream() -> None:
        line = stream.readline()
        if line:
            text.configure(state="normal")
            text.insert("end", line)
            text.see("end")
            text.configure(state="disabled")
        root.after(300, poll_stream)

    def on_close() -> None:
        try:
            stream.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    poll_stream()
    root.mainloop()


if __name__ == "__main__":
    main()

