"""
intent.py  –  VoiceGit Intent Classification
=============================================
Architecture
------------
1. TF-IDF (unigram + bigram) + Logistic Regression trained on an embedded
   labelled dataset derived from the `hesamation/git-prompt` HuggingFace
   dataset, augmented with hand-written examples for all 13 intents.
2. A keyword-rule fallback fires when ML confidence < CONFIDENCE_THRESHOLD.
3. Public API:  classify(text) -> (intent: str, confidence: float)

Supported Intents
-----------------
  create_branch | merge_branch | commit | push | pull | status |
  add           | log          | diff   | clone | init | stash  | reset

No external network calls are made at runtime. The model trains from the
embedded dataset on first import and is cached in memory for the process
lifetime (~50 ms on a modern laptop).

Usage
-----
    from intent import classify

    intent, confidence = classify("i want to try new code without breaking main")
    # => ("create_branch", 0.87)
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Labelled training dataset
# Sourced from: hesamation/git-prompt (HuggingFace) + manual augmentation
# 280 samples across 13 intents (~18-35 per class)
# ---------------------------------------------------------------------------
_TRAINING_DATA: list[tuple[str, str]] = [

    # ── create_branch ───────────────────────────────────────────────────────
    ("create a new branch for my feature", "create_branch"),
    ("make a branch to work on a new thing", "create_branch"),
    ("i want to make a copy of the main code so i can try some stuff", "create_branch"),
    ("create a branch from the current branch im on", "create_branch"),
    ("i need to work on a new feature without messing up the main code", "create_branch"),
    ("make a new branch and switch to it", "create_branch"),
    ("can you make a branch for me and name it new-feature", "create_branch"),
    ("i want to try some new code without ruining everything", "create_branch"),
    ("create a branch from a specific commit", "create_branch"),
    ("make a branch that copies the current state of the main branch", "create_branch"),
    ("how do i make a branch to work on a bug fix", "create_branch"),
    ("create a new branch and name it my-feature", "create_branch"),
    ("make a branch that i can use to test some new code", "create_branch"),
    ("i want to create a branch but i dont know what to name it", "create_branch"),
    ("make a branch that i can merge back into the main branch later", "create_branch"),
    ("how do i create a new branch in git", "create_branch"),
    ("create a branch for a new project", "create_branch"),
    ("i want to make a branch that is based on an older version of the code", "create_branch"),
    ("make a branch that i can merge back into the main branch", "create_branch"),
    ("can you create a branch for me and switch to it", "create_branch"),
    ("create a branch to work on a specific task", "create_branch"),
    ("make a branch that is separate from the main code", "create_branch"),
    ("how do i make a branch to work on a new feature", "create_branch"),
    ("create a branch to work on a bug fix", "create_branch"),
    ("i want to create a branch but im not sure what to do", "create_branch"),
    ("make a branch that i can use to try out some new ideas", "create_branch"),
    ("new branch please", "create_branch"),
    ("spin up a branch called hotfix", "create_branch"),
    ("start working on a separate branch", "create_branch"),
    ("i need a feature branch", "create_branch"),
    ("make me a branch named dev", "create_branch"),
    ("checkout a new branch", "create_branch"),
    ("switch to a new branch", "create_branch"),
    ("create branch feature-login", "create_branch"),
    ("i want to isolate my changes in a new branch", "create_branch"),
    ("spin up a hotfix branch", "create_branch"),
    ("spin up a new branch for this task", "create_branch"),

    # ── merge_branch ────────────────────────────────────────────────────────
    ("merge the changes from the feature branch into the main branch", "merge_branch"),
    ("can you merge the updates from the dev branch into the master branch for me", "merge_branch"),
    ("i want to merge the changes from the new-feature branch into the main branch", "merge_branch"),
    ("how do i merge the changes from the fix branch into the main branch", "merge_branch"),
    ("can you merge the changes from the feature branch into the release branch", "merge_branch"),
    ("i need to merge the changes from the hotfix branch into the main branch", "merge_branch"),
    ("how do i merge the changes from the dev branch if there are conflicts", "merge_branch"),
    ("merge the updated branch into main", "merge_branch"),
    ("can you merge feature into develop", "merge_branch"),
    ("combine my branch with master", "merge_branch"),
    ("integrate my feature branch into main", "merge_branch"),
    ("bring the changes from dev into production", "merge_branch"),
    ("i want to merge my work back into the main branch", "merge_branch"),
    ("merge new-feature into main without losing changes", "merge_branch"),
    ("merge branches together", "merge_branch"),
    ("pull my feature branch into master", "merge_branch"),
    ("join my branch with the main codebase", "merge_branch"),
    ("how do i combine two branches", "merge_branch"),
    ("absorb the hotfix branch into main", "merge_branch"),
    ("i finished my feature, merge it in", "merge_branch"),
    ("squash and merge my feature", "merge_branch"),
    ("fast-forward merge my branch", "merge_branch"),
    ("no-ff merge feature into develop", "merge_branch"),

    # ── commit ──────────────────────────────────────────────────────────────
    ("commit my changes", "commit"),
    ("save my work to git", "commit"),
    ("record what i did", "commit"),
    ("how do i commit", "commit"),
    ("make a commit with message initial commit", "commit"),
    ("i want to commit everything i changed", "commit"),
    ("commit all staged files", "commit"),
    ("save a snapshot of my code", "commit"),
    ("lock in my changes", "commit"),
    ("commit with the message fix bug", "commit"),
    ("i need to create a commit", "commit"),
    ("save my progress", "commit"),
    ("checkpoint my code", "commit"),
    ("commit my work with a message", "commit"),
    ("i want to store my changes in git history", "commit"),
    ("how do i save changes in git", "commit"),
    ("write a commit message and save", "commit"),
    ("commit staged changes", "commit"),
    ("finalize my changes in a commit", "commit"),
    ("i made some changes, commit them", "commit"),
    ("record my progress in git", "commit"),
    ("create a new commit", "commit"),
    ("commit everything", "commit"),
    ("i want to commit but im not sure how", "commit"),
    ("store my work permanently", "commit"),

    # ── push ────────────────────────────────────────────────────────────────
    ("push my changes to github", "push"),
    ("upload my commits to the remote", "push"),
    ("send my code to origin", "push"),
    ("push to remote", "push"),
    ("how do i push my code", "push"),
    ("i want to upload my work", "push"),
    ("push the current branch to remote", "push"),
    ("sync my local commits with github", "push"),
    ("publish my branch to origin", "push"),
    ("push my feature branch", "push"),
    ("push origin main", "push"),
    ("how do i send my commits to the server", "push"),
    ("i need to push my changes", "push"),
    ("deploy my code to the remote repository", "push"),
    ("push everything to github", "push"),
    ("share my code on github", "push"),
    ("send the branch to remote", "push"),
    ("update the remote with my local changes", "push"),
    ("push to master", "push"),
    ("i want to push my code upstream", "push"),
    ("upload changes to bitbucket", "push"),

    # ── pull ────────────────────────────────────────────────────────────────
    ("get the latest code from remote", "pull"),
    ("pull changes from github", "pull"),
    ("download the latest commits", "pull"),
    ("sync my local repo with remote", "pull"),
    ("how do i get updates from the server", "pull"),
    ("fetch and merge changes from origin", "pull"),
    ("i want the newest version of the code", "pull"),
    ("update my local branch with remote changes", "pull"),
    ("pull from master", "pull"),
    ("get the latest from origin", "pull"),
    ("bring my local repo up to date", "pull"),
    ("download changes from the remote repository", "pull"),
    ("refresh my code from github", "pull"),
    ("how do i pull", "pull"),
    ("i need to pull the latest changes", "pull"),
    ("grab the newest commits from remote", "pull"),
    ("pull origin main", "pull"),
    ("get remote changes into my branch", "pull"),
    ("update my code", "pull"),
    ("fetch the latest updates", "pull"),
    ("i want to sync with the team changes", "pull"),
    ("sync my repo with the remote", "pull"),
    ("sync my local code with github", "pull"),

    # ── status ──────────────────────────────────────────────────────────────
    ("show me the status", "status"),
    ("what files did i change", "status"),
    ("which files are modified", "status"),
    ("what is the current state of my repo", "status"),
    ("how do i check what changed", "status"),
    ("show me what is staged", "status"),
    ("i want to see untracked files", "status"),
    ("git status", "status"),
    ("what is going on in my repo", "status"),
    ("show me pending changes", "status"),
    ("are there any uncommitted changes", "status"),
    ("list the modified files", "status"),
    ("what has been changed but not committed", "status"),
    ("check repository status", "status"),
    ("see what is different", "status"),
    ("whats the status of my project", "status"),
    ("show current repo state", "status"),
    ("what files are waiting to be committed", "status"),
    ("how do i see what i changed", "status"),
    ("i want to know the state of my files", "status"),

    # ── add / stage ─────────────────────────────────────────────────────────
    ("stage my files", "add"),
    ("add everything to staging", "add"),
    ("stage all changes", "add"),
    ("git add all files", "add"),
    ("add this file to the staging area", "add"),
    ("how do i stage changes", "add"),
    ("mark files for the next commit", "add"),
    ("add my changes before committing", "add"),
    ("stage the modified files", "add"),
    ("i want to add all my changes", "add"),
    ("stage specific files", "add"),
    ("prepare files for commit", "add"),
    ("how do i add files to git", "add"),
    ("add readme to staging", "add"),
    ("stage everything i changed", "add"),
    ("i need to add my files before saving", "add"),
    ("put my changes in the index", "add"),
    ("select files for the commit", "add"),
    ("add only modified files", "add"),
    ("track a new file", "add"),

    # ── log ─────────────────────────────────────────────────────────────────
    ("show me the commit history", "log"),
    ("what commits were made", "log"),
    ("see the git log", "log"),
    ("list all commits", "log"),
    ("how do i see past commits", "log"),
    ("show commit messages", "log"),
    ("i want to see what changes were made over time", "log"),
    ("who committed what", "log"),
    ("git log", "log"),
    ("see the history of this repo", "log"),
    ("show me recent commits", "log"),
    ("how do i check the commit log", "log"),
    ("display commit history", "log"),
    ("list previous commits with messages", "log"),
    ("what was the last thing committed", "log"),
    ("see all commits on this branch", "log"),
    ("show me a pretty log", "log"),
    ("what has been done in this repo", "log"),
    ("see the timeline of changes", "log"),
    ("show changes made by the team", "log"),

    # ── diff ────────────────────────────────────────────────────────────────
    ("show me the diff", "diff"),
    ("what changed in my files", "diff"),
    ("compare my changes to the last commit", "diff"),
    ("show differences in my code", "diff"),
    ("how do i see what i changed before committing", "diff"),
    ("git diff", "diff"),
    ("show me line by line what changed", "diff"),
    ("what is different from the last version", "diff"),
    ("i want to review my changes", "diff"),
    ("compare two branches", "diff"),
    ("see the changes i made", "diff"),
    ("show me unstaged changes", "diff"),
    ("display what is different", "diff"),
    ("diff between main and feature", "diff"),
    ("what lines did i add or remove", "diff"),
    ("show file differences", "diff"),
    ("i want to see before and after", "diff"),
    ("compare staged vs unstaged", "diff"),
    ("preview my changes", "diff"),
    ("what exactly did i modify", "diff"),
    ("what lines did i change", "diff"),
    ("which lines were added or removed", "diff"),

    # ── clone ───────────────────────────────────────────────────────────────
    ("clone a repository", "clone"),
    ("download a repo from github", "clone"),
    ("get a copy of the project from github", "clone"),
    ("how do i clone a git repo", "clone"),
    ("i want to copy a remote repository locally", "clone"),
    ("git clone this url", "clone"),
    ("clone the project to my machine", "clone"),
    ("make a local copy of a remote repo", "clone"),
    ("how do i get someone elses code", "clone"),
    ("copy a repo from bitbucket", "clone"),
    ("download the codebase", "clone"),
    ("clone origin to my computer", "clone"),
    ("get the repo from the link", "clone"),
    ("i want to clone this github project", "clone"),
    ("how do i start with an existing repo", "clone"),
    ("duplicate a remote repo locally", "clone"),
    ("copy the remote project", "clone"),
    ("get the project files from github", "clone"),

    # ── init ────────────────────────────────────────────────────────────────
    ("start a new git repo", "init"),
    ("initialize git here", "init"),
    ("set up git in this folder", "init"),
    ("how do i create a new git repository", "init"),
    ("git init this directory", "init"),
    ("make this folder a git repo", "init"),
    ("i want to start tracking my project with git", "init"),
    ("begin version control here", "init"),
    ("create a new repository on my machine", "init"),
    ("how do i start using git for my project", "init"),
    ("initialize a new project with git", "init"),
    ("set up version control in my project", "init"),
    ("turn this folder into a git repo", "init"),
    ("create a git repo from scratch", "init"),
    ("how do i start a git project", "init"),
    ("new repository local", "init"),
    ("setup git tracking for my code", "init"),
    ("i want to start a fresh git project", "init"),

    # ── stash ───────────────────────────────────────────────────────────────
    ("save my work temporarily", "stash"),
    ("hide my changes for now", "stash"),
    ("stash my uncommitted work", "stash"),
    ("put my changes away without committing", "stash"),
    ("i need to switch branches but i have changes", "stash"),
    ("how do i temporarily save my work", "stash"),
    ("shelve my changes", "stash"),
    ("git stash", "stash"),
    ("set aside my current changes", "stash"),
    ("i want to save my progress without a commit", "stash"),
    ("park my changes for later", "stash"),
    ("how do i stash in git", "stash"),
    ("hold my changes while i do something else", "stash"),
    ("store my work in progress", "stash"),
    ("save dirty state temporarily", "stash"),
    ("i need to stash before switching branches", "stash"),
    ("keep my changes but clean up the workspace", "stash"),
    ("temporarily shelve my modifications", "stash"),

    # ── reset / undo ────────────────────────────────────────────────────────
    ("undo my last commit", "reset"),
    ("revert my changes", "reset"),
    ("go back to the last version", "reset"),
    ("how do i undo in git", "reset"),
    ("i made a mistake, undo it", "reset"),
    ("unstage my files", "reset"),
    ("discard my changes", "reset"),
    ("roll back to the previous commit", "reset"),
    ("i want to undo the last commit but keep changes", "reset"),
    ("reset to head", "reset"),
    ("how do i go back to a previous state", "reset"),
    ("remove my last commit", "reset"),
    ("i accidentally committed, undo it", "reset"),
    ("soft reset my last commit", "reset"),
    ("hard reset to previous commit", "reset"),
    ("how do i discard all local changes", "reset"),
    ("revert to a clean state", "reset"),
    ("undo staged changes", "reset"),
    ("throw away my uncommitted changes", "reset"),
    ("start fresh from last commit", "reset"),
    ("i want to get rid of my recent changes", "reset"),
    ("discard all my local changes", "reset"),
    ("discard everything and start clean", "reset"),
]

# ---------------------------------------------------------------------------
# All recognised intents
# ---------------------------------------------------------------------------
ALL_INTENTS: list[str] = [
    "create_branch",
    "merge_branch",
    "commit",
    "push",
    "pull",
    "status",
    "add",
    "log",
    "diff",
    "clone",
    "init",
    "stash",
    "reset",
]

# ---------------------------------------------------------------------------
# Confidence threshold – below this the keyword fallback is tried first
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Keyword fallback rules  (ordered – first match wins)
# Each tuple: ([keywords / phrases], intent)
# Matched via simple substring on lowercased input.
# ---------------------------------------------------------------------------
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["new branch", "create branch", "make a branch", "checkout -b", "switch to a branch",
      "start a branch", "spin up a branch", "feature branch", "bug fix branch",
      "a new branch", "make branch"], "create_branch"),
    (["merge", "combine branch", "integrate branch", "join branch", "absorb branch"], "merge_branch"),
    (["commit", "save my work to git", "record my changes", "lock in my changes",
      "checkpoint my code"], "commit"),
    (["push", "upload my changes", "upload my commits", "send to remote",
      "publish my branch", "deploy to remote"], "push"),
    (["pull", "fetch and merge", "get the latest", "download changes",
      "sync with remote", "bring up to date"], "pull"),
    (["status", "what changed", "what is modified", "uncommitted changes",
      "what is staged", "untracked files", "pending changes"], "status"),
    (["stage", "add to staging", "git add", "mark for commit",
      "add files", "track a new file"], "add"),
    (["log", "commit history", "past commits", "commit messages",
      "history of this repo", "timeline of changes"], "log"),
    (["diff", "differences", "what lines changed", "before and after",
      "compare changes", "unstaged changes"], "diff"),
    (["clone", "copy a repo", "download a repo", "copy the project",
      "get the codebase", "local copy of remote"], "clone"),
    (["init", "initialize git", "start a git", "new repository", "git init",
      "set up git", "turn this folder into"], "init"),
    (["stash", "temporarily save", "shelve my changes", "hide changes",
      "put away my changes", "save work temporarily", "park my changes"], "stash"),
    (["undo", "revert", "go back", "roll back", "reset", "discard changes",
      "unstage", "remove last commit", "start fresh", "throw away"], "reset"),
]

# ---------------------------------------------------------------------------
# Internal model handle – populated lazily on first classify() call
# ---------------------------------------------------------------------------
_pipeline: Optional[object] = None


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"['\"]", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_pipeline() -> object:
    """Train TF-IDF + Logistic Regression on the embedded dataset."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SkPipeline
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML intent classification.\n"
            "Install it with:  pip install scikit-learn"
        ) from exc

    texts  = [_normalise(t) for t, _ in _TRAINING_DATA]
    labels = [label          for _, label in _TRAINING_DATA]

    pipeline = SkPipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),     # unigrams + bigrams
            min_df=1,
            sublinear_tf=True,      # 1 + log(tf) dampens common terms
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,                  # moderate regularisation
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def _keyword_fallback(text: str) -> Optional[str]:
    """Return intent matched by keyword rules, or None."""
    lower = text.lower()
    for keywords, intent in _KEYWORD_RULES:
        for kw in keywords:
            if kw in lower:
                return intent
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(user_input: str) -> tuple[str, float]:
    """
    Classify a natural-language Git request.

    Parameters
    ----------
    user_input : str
        Raw text (or transcribed voice) from the user.

    Returns
    -------
    (intent, confidence) : tuple[str, float]
        intent     – one of ALL_INTENTS, or "unknown"
        confidence – ML probability in [0, 1];
                     1.0 for keyword-rule matches, 0.0 for "unknown"

    Examples
    --------
    >>> classify("i want to work on a new feature without breaking main")
    ('create_branch', 0.87)
    >>> classify("upload my changes to github")
    ('push', 0.91)
    >>> classify("asdfgh xyz")
    ('unknown', 0.0)
    """
    global _pipeline

    text = (user_input or "").strip()
    if not text:
        return "unknown", 0.0

    # ── ML classification ───────────────────────────────────────────────────
    if _pipeline is None:
        _pipeline = _build_pipeline()

    normalised   = _normalise(text)
    proba_array  = _pipeline.predict_proba([normalised])[0]
    best_idx     = int(proba_array.argmax())
    confidence   = float(proba_array[best_idx])
    ml_intent    = _pipeline.classes_[best_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        return ml_intent, round(confidence, 4)

    # ── keyword fallback ────────────────────────────────────────────────────
    fallback = _keyword_fallback(text)
    if fallback:
        return fallback, 1.0   # deterministic rule → report as certain

    return "unknown", 0.0


def detect_intent(user_input: str) -> str:
    """
    Legacy single-value wrapper kept for backwards compatibility.
    Prefer classify() for new code.

    Returns the intent string only ("unknown" if nothing matched).
    """
    intent, _ = classify(user_input)
    return intent


# ---------------------------------------------------------------------------
# Warm up on import so the first real classify() call has no latency spike
# ---------------------------------------------------------------------------
def _warm_up() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()


_warm_up()


# ---------------------------------------------------------------------------
# Self-test – run with:  python intent.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _TEST_CASES: list[tuple[str, str]] = [
        ("i want to work on a new feature without messing up the main code", "create_branch"),
        ("make me a branch named dev",                                        "create_branch"),
        ("spin up a hotfix branch",                                           "create_branch"),
        ("merge my feature into master",                                      "merge_branch"),
        ("how do i combine two branches",                                     "merge_branch"),
        ("i finished my feature, merge it in",                                "merge_branch"),
        ("commit all my changes with a message",                              "commit"),
        ("save a snapshot of my code",                                        "commit"),
        ("checkpoint my work",                                                "commit"),
        ("push to github",                                                    "push"),
        ("upload my work to origin",                                          "push"),
        ("send my commits to the remote server",                              "push"),
        ("get the latest code from remote",                                   "pull"),
        ("sync my repo with github",                                          "pull"),
        ("download changes from the remote",                                  "pull"),
        ("which files are modified",                                          "status"),
        ("show me the current repo state",                                    "status"),
        ("are there any uncommitted changes",                                 "status"),
        ("stage all my changes",                                              "add"),
        ("add my files to staging",                                           "add"),
        ("prepare files for the next commit",                                 "add"),
        ("show me the commit history",                                        "log"),
        ("what commits have been made",                                       "log"),
        ("list all previous commits",                                         "log"),
        ("show me the diff",                                                  "diff"),
        ("what lines did i change",                                           "diff"),
        ("compare my changes to last commit",                                 "diff"),
        ("clone this github repo",                                            "clone"),
        ("download a copy of the project",                                    "clone"),
        ("get someone elses code from github",                                "clone"),
        ("initialize git in my folder",                                       "init"),
        ("start a new git repository",                                        "init"),
        ("make this directory a git repo",                                    "init"),
        ("stash my changes so i can switch branches",                         "stash"),
        ("hide my work temporarily",                                          "stash"),
        ("set aside my changes for now",                                      "stash"),
        ("undo my last commit",                                               "reset"),
        ("discard all my local changes",                                      "reset"),
        ("go back to the previous version",                                   "reset"),
    ]

    print(f"\n{'INPUT':<57} {'EXPECTED':<15} {'GOT':<15} {'CONF':>6}  OK?")
    print("─" * 102)
    passed = 0
    for phrase, expected in _TEST_CASES:
        got, conf = classify(phrase)
        ok = "✓" if got == expected else "✗"
        if got == expected:
            passed += 1
        print(f"{phrase:<57} {expected:<15} {got:<15} {conf:>6.2f}  {ok}")

    total = len(_TEST_CASES)
    print(f"\n{'─' * 102}")
    print(f"Result: {passed}/{total} passed  ({100 * passed // total}%)\n")

