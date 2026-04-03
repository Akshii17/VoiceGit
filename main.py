import sys
from typing import List

import learning
from commands import generate_commands
from executor import execute_commands
from intent import ALL_INTENTS, classify
from learning import (
    explain_command,
    start_learning_window,
    stop_learning_window,
    store_learning,
    write_to_learning_stream,
)
from safety import validate_commands
from state import get_repo_state
from voice import listen_and_transcribe


SUPPORTED_COMMANDS = {"help", "exit", "quit"}
voice_mode = False

CLASSIFY_MIN_CONFIDENCE = 0.35


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_error(message: str) -> None:
    print(f"\nError: {message}", file=sys.stderr)


def print_suggested_commands(commands: List[str]) -> None:
    print_header("suggested commands")
    for c in commands:
        print(f"- {c}")


def confirm_proceed() -> bool:
    while True:
        try:
            answer = input("Proceed? (y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print("Please enter 'y' or 'n'.")


def process_learning(commands: List[str]) -> None:
    for cmd in commands:
        if cmd.startswith("ERROR"):
            continue

        explanation = explain_command(cmd)
        if not explanation:
            continue

        store_learning(cmd, explanation)
        if learning.learning_mode:
            write_to_learning_stream(f"{cmd} -> {explanation}")


def show_help() -> None:
    print("Supported commands:")
    print("  Natural language (ML): create_branch, merge_branch, commit, push,")
    print("    pull, status, add, log, diff, clone, init, stash, reset")
    print("  state   - show parsed repository state")
    print("  voice on / voice off - voice input toggle")
    print("  learn / close learn - learning window")
    print("  help / exit")


def handle_command_text(user_cmd: str) -> bool:
    """
    Handle one user command.
    Returns True to continue loop, False to exit.
    """
    global voice_mode

    cmd_lower = user_cmd.lower()
    if cmd_lower in {"exit", "quit", "q"}:
        stop_learning_window()
        print("Bye.")
        return False

    if cmd_lower == "help":
        show_help()
        return True

    if cmd_lower == "learn":
        try:
            start_learning_window()
            print("Learning mode enabled.")
        except Exception as e:
            print_error(f"Could not start learning window: {e}")
        return True

    if cmd_lower == "close learn":
        stop_learning_window()
        print("Learning mode disabled.")
        return True

    if cmd_lower == "voice on":
        voice_mode = True
        print("Voice mode enabled.")
        return True

    if cmd_lower == "voice off":
        voice_mode = False
        print("Voice mode disabled.")
        return True

    if cmd_lower == "state":
        try:
            state = get_repo_state()
            print_header("repo state")
            print(f"current_branch:   {state.current_branch}")
            print(f"has_commits:      {state.has_commits}")
            print(f"has_changes:      {state.has_changes}")
            print(f"staged_files:     {state.staged_files}")
            print(f"ahead_of_remote:  {state.ahead_of_remote}")
            print(f"behind_of_remote: {state.behind_of_remote}")
            print(f"has_conflicts:    {state.has_conflicts}")
        except RuntimeError as e:
            print_error(str(e))
        except Exception as e:
            print_error(f"Unexpected error: {e}")
        return True

    intent, confidence = classify(user_cmd)

    print(f"Input: {user_cmd}")
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence}")
    print(f"Detected intent: {intent} (confidence: {confidence})")

    if confidence < CLASSIFY_MIN_CONFIDENCE or intent == "unknown":
        print("Could not understand confidently. Try rephrasing.")
        return True

    if intent not in ALL_INTENTS:
        print_error(f"Intent '{intent}' is not supported.")
        return True

    try:
        if intent in {"init", "clone"}:
            repo_state = None
        else:
            repo_state = get_repo_state()

        suggested = generate_commands(intent, repo_state)
        print_suggested_commands(suggested)

        ok, reason = validate_commands(suggested, repo_state)
        if not ok:
            print_error(f"Unsafe to execute: {reason}")
            return True

        if not confirm_proceed():
            print("Cancelled.")
            return True

        process_learning(suggested)
        execute_commands(suggested)
    except RuntimeError as e:
        print_error(str(e))
    except Exception as e:
        print_error(f"Unexpected error: {e}")
    return True


def main() -> None:
    global voice_mode

    print("voicegit: a tiny interactive git helper")
    print("Type `help` for options.")

    while True:
        user_cmd: str = ""
        if voice_mode:
            try:
                transcribed = listen_and_transcribe()
                if transcribed is None:
                    continue
                user_cmd = transcribed.strip()
            except RuntimeError as e:
                print_error(str(e))
                print("Falling back to text input.")
                try:
                    user_cmd = input("Enter command: ").strip()
                except (EOFError, KeyboardInterrupt):
                    stop_learning_window()
                    print("\nBye.")
                    return
            except (EOFError, KeyboardInterrupt):
                stop_learning_window()
                print("\nBye.")
                return
        else:
            try:
                user_cmd = input("Enter command: ").strip()
            except (EOFError, KeyboardInterrupt):
                stop_learning_window()
                print("\nBye.")
                return

        if not user_cmd:
            continue

        if not handle_command_text(user_cmd):
            return


if __name__ == "__main__":
    main()

