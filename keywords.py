from __future__ import annotations

KEYWORD_RULES: list[tuple[list[str], str]] = [

    # ── reset variants  (BEFORE reset) ───────────────────────────────────────
    (["reset --soft", "soft reset", "undo commit keep staged",
      "uncommit but keep staged", "keep changes staged",
      "undo the commit and leave changes staged",
      "take back commit keep staged"], "reset_soft"),

    (["reset --hard", "hard reset", "nuke everything",
      "discard all changes and go back",
      "throw away all my local changes completely",
      "wipe all local changes", "lose all my uncommitted work",
      "discard working directory and staging area"], "reset_hard"),

    (["reset head file", "unstage a specific file",
      "remove a file from the staging area",
      "undo git add for one file", "take a file out of staging",
      "accidentally staged a file", "unstage readme",
      "pull a file back out of the index",
      "undo staging for one file", "remove from staging"], "reset_unstage"),

    (["mixed reset", "undo commit and unstage",
      "uncommit and unstage", "unstage and uncommit",
      "reset head~1 without flag", "go back one commit and unstage"], "reset_mixed"),

    # ── stash variants  (BEFORE stash) ───────────────────────────────────────
    (["stash pop", "pop the stash", "apply stash",
      "unstash my changes", "restore stashed",
      "get stashed changes back", "retrieve stash",
      "apply what i stashed", "bring back stashed",
      "continue stashed work", "apply the stash and remove",
      "apply last stash entry"], "stash_pop"),

    (["stash apply stash@", "apply a specific stash", "apply stash number",
      "apply older stash", "apply stash without removing",
      "use stash but keep it", "apply stash at index",
      "restore stash without dropping", "reapply stash",
      "apply named stash", "stash apply to use without popping"], "stash_apply"),

    (["stash list", "list all stashes", "show all stashes",
      "what stashes do i have", "show all stash entries",
      "display the stash stack", "view stash history",
      "list the stash stack", "what is in my stash",
      "how many stashes have i saved"], "stash_list"),

    (["stash drop", "delete a stash", "drop the stash",
      "remove a stash", "clear stash", "discard a stash",
      "get rid of stash", "remove stash at index",
      "delete old stash", "clean up stashes"], "stash_drop"),

    # ── log variants  (BEFORE log – "commit log" contains "log") ─────────────
    (["commit history", "commit log", "commit hostory", "git log --oneline",
      "what all has happened", "what has happened", "history of this repo",
      "project history", "timeline of changes", "past commits",
      "previous commits", "recent commits", "see the history",
      "what has been done", "happened so far", "full history",
      "show changes made by the team", "what happened in this project"], "log"),

    (["log --oneline", "one line per commit", "compact log",
      "compact commit history", "abbreviated commit", "brief commit",
      "condensed commit", "short log", "minimal log",
      "show commit titles", "just commit messages"], "log_oneline"),

    (["log --graph", "commit graph", "branch graph", "commit tree",
      "visual history", "branch topology", "branch lines",
      "draw commit graph", "graph of branches",
      "branches visually", "merges and branches visually"], "log_graph"),

    (["log for one file", "history of a file", "commits for a file",
      "commits that changed a file", "log for a specific file",
      "which commits modified this file", "what happened to this file",
      "file-specific commit history", "track history of one file",
      "log file history", "changes to a particular file"], "log_file"),

    (["log --stat", "log with stats", "log with statistics",
      "lines changed per commit", "insertions and deletions",
      "commit stats", "size of each commit",
      "changed files per commit", "diff stat per commit",
      "how much changed in each commit"], "log_stat"),

    # ── diff variants  (BEFORE diff) ─────────────────────────────────────────
    (["diff --staged", "diff --cached", "staged diff",
      "what is staged for commit", "what will be in my next commit",
      "what will i commit", "preview what will be committed",
      "what have i staged", "show staged changes",
      "staged vs last commit", "diff of staged",
      "what exactly will i commit"], "diff_staged"),

    (["diff between branches", "diff two branches", "compare two branches",
      "diff branch1 branch2", "compare feature branch",
      "what is different between my branch and main",
      "compare branches", "see differences between branches",
      "diff my branch against", "how does my branch differ"], "diff_branches"),

    (["diff head~", "diff commit", "compare to a specific commit",
      "diff since commit", "compare to an older commit",
      "what changed between commits", "changes since last release",
      "diff between two commit hashes", "compare commits"], "diff_commit"),

    # ── rebase variants  (BEFORE rebase) ──────────────────────────────────────
    (["rebase --abort", "abort the rebase", "cancel the rebase",
      "stop the rebase", "bail out of rebase",
      "quit the rebase", "undo the rebase",
      "exit rebase with conflicts", "back out of rebase"], "rebase_abort"),

    (["rebase --continue", "continue the rebase", "resume rebase",
      "proceed with rebase", "finish the rebase",
      "move to next step in rebase", "fixed conflicts continue rebase",
      "keep going with rebase"], "rebase_continue"),

    (["rebase -i", "interactive rebase", "squash commits",
      "squash my commits", "squash all commits into one",
      "clean up commit history with rebase",
      "rewrite commit history", "edit commit messages rebase",
      "combine commits into one", "reorder my commits",
      "drop some commits", "squash last", "flatten commits"], "rebase_interactive"),

    # ── merge variants  (BEFORE merge) ────────────────────────────────────────
    (["merge --abort", "abort the merge", "cancel the merge",
      "stop the merge", "undo merge in progress",
      "exit merge with conflicts", "back out of merge",
      "cancel merge and restore", "dont want to finish this merge"], "merge_abort"),

    (["merge --squash", "squash and merge", "squash merge",
      "merge with squash", "squash commits then merge",
      "flatten commits then merge", "single squashed commit",
      "squash into one before merging"], "merge_squash"),

    # ── push variants  (BEFORE push) ─────────────────────────────────────────
    (["push --force", "push -f", "push force", "force push",
      "force-with-lease", "overwrite remote with local",
      "push after rebase", "push even though histories diverged",
      "force update remote branch"], "push_force"),

    (["push -u", "push --set-upstream", "set upstream and push",
      "push and track", "first time pushing this branch",
      "push new branch to remote", "create remote tracking branch",
      "push branch that doesnt exist on remote",
      "push and register remote branch",
      "link local branch to remote when pushing"], "push_upstream"),

    (["push --tags", "push tags", "push all tags",
      "upload tags", "push git tags", "push version tags",
      "push annotated tags", "push release tags",
      "send tags to remote", "share tags with team"], "push_tags"),

    # ── clone variants  (BEFORE clone) ────────────────────────────────────────
    (["clone --depth", "shallow clone", "clone without full history",
      "fast clone without commits", "clone latest snapshot",
      "clone with depth", "reduce clone size",
      "clone minimal history", "truncated clone",
      "partial clone", "clone just the tip"], "clone_shallow"),

    (["clone -b", "clone a branch", "clone specific branch",
      "clone only one branch", "clone the dev branch",
      "clone and checkout branch", "clone targeting branch",
      "clone on the branch", "clone just one branch"], "clone_branch"),

    # ── remote variants  (BEFORE remote) ──────────────────────────────────────
    (["remote add", "add a remote", "add remote origin",
      "set remote origin", "connect repo to github",
      "add upstream remote", "link local repo to remote",
      "point repo at github", "configure remote origin",
      "register remote", "set up a remote"], "remote_add"),

    (["remote remove", "remote rm", "delete remote",
      "remove the remote", "unlink from remote",
      "delete upstream remote", "remove origin",
      "unset the remote", "disconnect from remote",
      "get rid of remote", "detach from remote",
      "discard remote configuration"], "remote_remove"),

    # ── pull variant ──────────────────────────────────────────────────────────
    (["pull --rebase", "pull with rebase", "pull rebase",
      "fetch and rebase", "pull using rebase",
      "pull without merge commit", "linear history pull",
      "pull rebase to avoid merge commits",
      "sync using pull rebase", "use rebase when pulling"], "pull_rebase"),

    # ── commit_amend  (BEFORE commit) ─────────────────────────────────────────
    (["commit --amend", "amend commit", "amend my last commit",
      "edit last commit", "fix commit message",
      "fix the commit message", "fix my commit",
      "change commit message", "forgot to include a file in commit",
      "add file to last commit", "modify last commit",
      "update last commit", "forgot something in last commit",
      "append to last commit", "rewrite last commit"], "commit_amend"),

    # ── restore  (BEFORE reset – more specific single-file undo) ──────────────
    (["restore a file", "restore this file", "git restore",
      "undo changes to a file", "discard changes in a file",
      "discard edits in", "put this file back",
      "undo just that file", "revert a file",
      "reset one file to last commit", "put file back to original",
      "bring file back to committed"], "restore"),

    # ── revert  (BEFORE reset – different semantics) ───────────────────────────
    (["git revert", "revert head", "revert the last commit",
      "undo a pushed commit", "create a reverting commit",
      "safe undo", "undo without rewriting history",
      "make an undo commit", "already pushed how do i undo",
      "undo commit by making new commit",
      "roll back with a new commit"], "revert"),

    # ── reset ──────────────────────────────────────────────────────────────────
    (["git reset", "reset to", "reset head",
      "undo my last commit", "remove last commit",
      "undo commit", "take back commit",
      "go back to previous state", "undo in git",
      "throw away uncommitted", "discard all local"], "reset"),

    # ── stash  (AFTER stash variants) ─────────────────────────────────────────
    (["git stash", "stash my", "stash changes",
      "stash before switching", "temporarily save",
      "shelve my changes", "hide changes",
      "put away my changes", "park my changes",
      "save work temporarily", "set aside changes"], "stash"),

    # ── delete_branch  (BEFORE list/create) ────────────────────────────────────
    (["delete a branch", "delete branch", "delete the branch",
      "delete my branch", "delete this branch",
      "delete the feature branch", "delete the hotfix branch",
      "remove a branch", "remove branch", "remove the branch",
      "branch -d", "branch -D", "clean up branch",
      "wipe this branch", "prune old branches",
      "no longer need this branch", "dont need this branch",
      "i dont need this branch", "get rid of a branch",
      "delete merged branch", "finished with branch"], "delete_branch"),

    # ── list_branches ─────────────────────────────────────────────────────────
    (["all the branches", "all branches", "list branches", "show branches",
      "what branches", "which branches", "available branches",
      "current branch", "what branch am i", "which branch am i",
      "branch am i on", "my current branch", "local and remote branches",
      "branch -a", "show remote branches", "see all branches"], "list_branches"),

    # ── create_branch ─────────────────────────────────────────────────────────
    (["new branch", "create branch", "make a branch", "checkout -b",
      "switch -c", "a new branch", "make branch", "spin up a branch",
      "bug fix branch", "start a branch",
      "isolate my changes", "create and switch to a new branch"], "create_branch"),

    # ── checkout  (AFTER create_branch) ───────────────────────────────────────
    ("switch","switch branch", "switch to branch" "switch to main", "switch to master", "switch to develop",
      "go to branch", "checkout main", "checkout master",
      "git switch ", "git checkout ", "change to the branch",
      "move to another branch", "navigate to branch",
      "get on the main branch", "jump to master",
      "go to the branch", "i want to be on branch", "checkout"),

    # ── rebase  (AFTER rebase variants) ───────────────────────────────────────
    (["git rebase", "rebase onto", "rebase my branch",
      "rebase instead of merge", "apply commits on top",
      "keep branch up to date with rebase",
      "integrate upstream with rebase",
      "linear history using rebase"], "rebase"),

    # ── merge_branch  (AFTER merge variants) ──────────────────────────────────
    (["merge", "combine branch", "integrate branch",
      "join branch", "absorb branch", "bring branch into",
      "merge my branch"], "merge_branch"),

    # ── fetch ─────────────────────────────────────────────────────────────────
    (["git fetch", "fetch origin", "fetch all", "fetch remote",
      "fetch without merging", "dont merge yet", "just fetch",
      "download remote changes without", "remote tracking branches",
      "preview remote changes", "check what changed remotely"], "fetch"),

    # ── diff  (AFTER diff variants) ────────────────────────────────────────────
    (["show me the diff", "git diff", "what lines changed",
      "before and after", "file differences", "compare changes",
      "unstaged changes", "diff between",
      "what lines did i", "show differences", "what changed in",
      "what is different from"], "diff"),

    # ── status ────────────────────────────────────────────────────────────────
    (["git status", "repo status", "what changed", "what is modified",
      "uncommitted changes", "what is staged", "untracked files",
      "pending changes", "modified files", "show me the status",
      "state of my files", "what files did i",
      "what has been changed", "status"], "status"),

    # ── add_patch  (BEFORE add) ────────────────────────────────────────────────
    (["add -p", "add patch", "interactive staging", "interactive add",
      "stage partial", "stage only part", "stage hunks",
      "stage piece by piece", "choose what to stage",
      "review each change before staging",
      "selectively stage", "add selected hunks"], "add_patch"),

    # ── add ────────────────────────────────────────────────────────────────────
    (["stage", "add to staging", "git add", "mark for commit",
      "add files", "track a new file", "staging area",
      "add all files", "add everything", "stage my changes",
      "put in index", "prepare for commit"], "add"),

    # ── commit  (AFTER commit_amend) ───────────────────────────────────────────
    (["commit my", "make a commit", "git commit", "create a commit",
      "save a snapshot", "lock in my changes", "commit staged",
      "commit everything", "commit with the message",
      "commit all my changes", "commit all staged",
      "record my changes", "store my work", "commit"], "commit"),

    # ── push  (AFTER push variants) ────────────────────────────────────────────
    (["push", "upload my changes", "upload my commits",
      "send to remote", "publish my branch",
      "deploy to remote", "push to", "push origin",
      "send my code", "share my code"], "push"),

    # ── pull  (AFTER pull_rebase) ───────────────────────────────────────────────
    (["pull", "fetch and merge", "get the latest", "download changes",
      "sync with remote", "bring up to date",
      "sync my local", "sync my repo", "pull origin",
      "update my local branch", "get remote changes"], "pull"),

    # ── clone  (AFTER clone variants) ──────────────────────────────────────────
    (["clone", "copy a repo", "download a repo", "copy the project",
      "get the codebase", "local copy of remote",
      "get someone elses code"], "clone"),

    # ── init ───────────────────────────────────────────────────────────────────
    (["init", "initialize git", "start a git", "new repository",
      "git init", "set up git", "turn this folder into",
      "start tracking my project", "version control here"], "init"),

    # ── remote  (AFTER remote variants) ───────────────────────────────────────
    (["remote -v", "remote url", "remote origin", "remote connection",
      "show remotes", "list remotes", "manage remote",
      "what remotes", "my remotes", "show me my remote",
      "view origin", "check remote address", "see my remotes"], "remote"),

    # ── show ──────────────────────────────────────────────────────────────────
    (["git show", "show commit", "inspect a commit", "commit details",
      "what did that commit", "what was in that commit",
      "show me a commit", "view commit", "commit content",
      "full diff of a commit", "changes from the last commit"], "show"),

    # ── blame ─────────────────────────────────────────────────────────────────
    (["blame", "git blame", "who wrote", "who changed",
      "who committed this line", "per line author",
      "annotate with author", "who last edited",
      "who is responsible for this code"], "blame"),

    # ── remove_file ───────────────────────────────────────────────────────────
    (["git rm", "remove a file", "delete a file from git",
      "untrack a file", "stop tracking", "remove tracked file",
      "remove from repo", "delete from git",
      "remove file from index"], "remove_file"),

    # ── move_file ─────────────────────────────────────────────────────────────
    (["git mv", "rename a file", "move a file", "rename this file",
      "move file to", "git move", "git rename",
      "rename tracked file", "rename and track",
      "change name of tracked file"], "move_file"),
]
