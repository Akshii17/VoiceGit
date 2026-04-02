from __future__ import annotations


def detect_intent(user_input: str) -> str:
    """
    Detect a simple git-related intent from free-form user text.

    Supported intents:
      - status
      - add
      - commit
      - push
      - create_branch

    Returns:
      - one of the supported intents, or "unknown"
    """
    text = (user_input or "").strip().lower()
    if not text:
        return "unknown"

    # Phrase-based checks first (more specific).
    if "create branch" in text or "new branch" in text:
        return "create_branch"

    # Keyword matching.
    if "status" in text:
        return "status"

    if "add" in text or "stage" in text:
        return "add"

    if "commit" in text or "save" in text:
        return "commit"

    if "push" in text or "upload" in text:
        return "push"

    return "unknown"

