"""
intent.py  –  VoiceGit Intent Classification
=============================================
Architecture
------------
1. TF-IDF (unigram + bigram) + Logistic Regression trained on ~950 embedded
   labelled examples covering 50 Git intents.
   Dataset derived from hesamation/git-prompt (HuggingFace) + manual
   augmentation for every popularly-used git command and its variants.
2. Keyword-rule fallback fires when ML confidence < CONFIDENCE_THRESHOLD.
   Rules use longest-match-first ordering to prevent short tokens (e.g.
   "reset") from swallowing longer phrases ("reset --soft", "reset --hard").
3. Public API:  classify(text) -> (intent: str, confidence: float)

Supported Intents  (50 total)
------------------------------
  Setup:         init | clone | clone_branch | clone_shallow | remote
                 remote_add | remote_remove

  Core workflow: status | add | add_patch | commit | commit_amend
                 push | push_force | push_upstream | push_tags
                 pull | pull_rebase | fetch

  Branching:     list_branches | create_branch | checkout
                 delete_branch | merge_branch | merge_abort | merge_squash
                 rebase | rebase_interactive | rebase_abort | rebase_continue

  Inspection:    log | log_oneline | log_graph | log_file | log_stat
                 diff | diff_staged | diff_branches | diff_commit
                 show | blame

  Undoing:       restore | revert | reset | reset_soft | reset_hard
                 reset_mixed | reset_unstage
                 stash | stash_pop | stash_apply | stash_list | stash_drop

  File ops:      remove_file | move_file

No external network calls are made at runtime. The model trains from the
embedded dataset on first import (~100 ms on a modern laptop).

Usage
-----
    from intent import classify, detect_intent, ALL_INTENTS

    intent, confidence = classify("undo my last commit but keep my changes")
    # => ("reset_soft", 0.84)

    intent, confidence = classify("squash all my commits into one")
    # => ("rebase_interactive", 0.79)
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Labelled training dataset  (~950 examples, 50 intents)
# ---------------------------------------------------------------------------
_TRAINING_DATA: list[tuple[str, str]] = [

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP & CONFIG
    # ══════════════════════════════════════════════════════════════════════════

    # ── init ─────────────────────────────────────────────────────────────────
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
    ("new local repository please", "init"),
    ("setup git tracking for my code", "init"),
    ("i want to start a fresh git project", "init"),
    ("make this directory version controlled", "init"),
    ("start tracking this project", "init"),

    # ── clone ─────────────────────────────────────────────────────────────────
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
    ("download the codebase from github", "clone"),
    ("clone origin to my computer", "clone"),
    ("get the repo from the link", "clone"),
    ("i want to clone this github project", "clone"),
    ("how do i start with an existing repo", "clone"),
    ("duplicate a remote repo locally", "clone"),
    ("copy the remote project to my machine", "clone"),
    ("get the project files from github", "clone"),
    ("grab the source code from gitlab", "clone"),
    ("bring a remote repo to my local machine", "clone"),

    # ── clone_branch ──────────────────────────────────────────────────────────
    ("clone a specific branch", "clone_branch"),
    ("git clone -b develop", "clone_branch"),
    ("clone only the main branch", "clone_branch"),
    ("how do i clone a specific branch from github", "clone_branch"),
    ("i want to clone just one branch not the whole repo", "clone_branch"),
    ("clone the dev branch of this repo", "clone_branch"),
    ("download only the release branch", "clone_branch"),
    ("clone a repo and check out a specific branch", "clone_branch"),
    ("git clone branch flag", "clone_branch"),
    ("clone with a specific branch selected", "clone_branch"),
    ("how do i get only one branch from remote", "clone_branch"),
    ("clone and checkout feature branch", "clone_branch"),
    ("clone the repository on the staging branch", "clone_branch"),
    ("i only want to clone the feature branch", "clone_branch"),
    ("clone targeting a particular branch", "clone_branch"),

    # ── clone_shallow ─────────────────────────────────────────────────────────
    ("shallow clone the repo", "clone_shallow"),
    ("git clone --depth 1", "clone_shallow"),
    ("clone without the full history", "clone_shallow"),
    ("i want a fast clone without all commits", "clone_shallow"),
    ("how do i clone just the latest snapshot", "clone_shallow"),
    ("clone with depth 1", "clone_shallow"),
    ("do a shallow clone to save space", "clone_shallow"),
    ("clone only the most recent commit", "clone_shallow"),
    ("i dont need the full history just the latest code", "clone_shallow"),
    ("how do i reduce clone size", "clone_shallow"),
    ("clone faster by limiting history depth", "clone_shallow"),
    ("truncated clone", "clone_shallow"),
    ("clone with minimal history", "clone_shallow"),
    ("partial clone of the repo", "clone_shallow"),
    ("clone just the tip of the branch", "clone_shallow"),

    # ── remote ────────────────────────────────────────────────────────────────
    ("show me the remote repositories", "remote"),
    ("what remotes do i have", "remote"),
    ("list my remote connections", "remote"),
    ("show remote origin url", "remote"),
    ("how do i see my remote", "remote"),
    ("git remote -v", "remote"),
    ("what is my remote url", "remote"),
    ("show all remotes", "remote"),
    ("what remote is this repo connected to", "remote"),
    ("check the remote address", "remote"),
    ("view origin url", "remote"),
    ("show remote tracking branches", "remote"),
    ("manage remote connections", "remote"),
    ("list all remotes", "remote"),
    ("what remotes are configured", "remote"),

    # ── remote_add ────────────────────────────────────────────────────────────
    ("add a remote repository", "remote_add"),
    ("set the remote origin", "remote_add"),
    ("how do i connect to a remote", "remote_add"),
    ("add origin to my repo", "remote_add"),
    ("git remote add origin", "remote_add"),
    ("how do i add an upstream remote", "remote_add"),
    ("connect my repo to github", "remote_add"),
    ("set up a remote for my project", "remote_add"),
    ("add a new remote called upstream", "remote_add"),
    ("link my local repo to a remote", "remote_add"),
    ("how do i point my repo at github", "remote_add"),
    ("add remote origin url", "remote_add"),
    ("i want to add a remote to my project", "remote_add"),
    ("register a remote repository", "remote_add"),
    ("configure remote origin", "remote_add"),

    # ── remote_remove ─────────────────────────────────────────────────────────
    ("remove a remote", "remote_remove"),
    ("delete the remote connection", "remote_remove"),
    ("git remote remove origin", "remote_remove"),
    ("how do i unlink my repo from remote", "remote_remove"),
    ("i want to delete the upstream remote", "remote_remove"),
    ("remove the origin remote", "remote_remove"),
    ("unset the remote", "remote_remove"),
    ("git remote rm origin", "remote_remove"),
    ("how do i disconnect from a remote", "remote_remove"),
    ("get rid of a remote", "remote_remove"),
    ("delete a remote connection", "remote_remove"),
    ("remove the remote called origin", "remote_remove"),
    ("how do i remove remote tracking", "remote_remove"),
    ("detach from remote", "remote_remove"),
    ("discard the remote configuration", "remote_remove"),

    # ══════════════════════════════════════════════════════════════════════════
    # CORE WORKFLOW
    # ══════════════════════════════════════════════════════════════════════════

    # ── status ────────────────────────────────────────────────────────────────
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
    ("whats the status of my project", "status"),
    ("show current repo state", "status"),
    ("what files are waiting to be committed", "status"),
    ("how do i see what i changed", "status"),
    ("i want to know the state of my files", "status"),
    ("show me what is different in my working directory", "status"),
    ("what files have i modified", "status"),
    ("tell me what is going on", "status"),

    # ── add ───────────────────────────────────────────────────────────────────
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
    ("git add dot", "add"),
    ("add all files to git", "add"),
    ("stage my whole project", "add"),

    # ── add_patch ─────────────────────────────────────────────────────────────
    ("add changes interactively", "add_patch"),
    ("stage only part of my changes", "add_patch"),
    ("i want to stage specific hunks", "add_patch"),
    ("git add -p", "add_patch"),
    ("git add patch mode", "add_patch"),
    ("interactive staging", "add_patch"),
    ("let me choose which changes to stage", "add_patch"),
    ("stage changes piece by piece", "add_patch"),
    ("i want to review each change before staging", "add_patch"),
    ("how do i stage partial changes", "add_patch"),
    ("stage only some of my edits", "add_patch"),
    ("pick which parts of a file to stage", "add_patch"),
    ("i dont want to stage all my changes at once", "add_patch"),
    ("add selected hunks to staging", "add_patch"),
    ("selectively stage my modifications", "add_patch"),
    ("add patch", "add_patch"),
    ("interactive add", "add_patch"),
    ("choose what to stage line by line", "add_patch"),

    # ── commit ────────────────────────────────────────────────────────────────
    ("commit my changes", "commit"),
    ("save my work to git", "commit"),
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
    ("create a new commit", "commit"),
    ("commit everything", "commit"),
    ("i want to commit but im not sure how", "commit"),
    ("store my work permanently", "commit"),
    ("make a git commit", "commit"),
    ("commit with a descriptive message", "commit"),

    # ── commit_amend ──────────────────────────────────────────────────────────
    ("amend my last commit", "commit_amend"),
    ("edit the last commit message", "commit_amend"),
    ("git commit --amend", "commit_amend"),
    ("i made a typo in my commit message", "commit_amend"),
    ("how do i fix my last commit message", "commit_amend"),
    ("change the message of my last commit", "commit_amend"),
    ("add more changes to my last commit", "commit_amend"),
    ("i forgot to include a file in my last commit", "commit_amend"),
    ("update my last commit without creating a new one", "commit_amend"),
    ("rewrite the last commit", "commit_amend"),
    ("how do i add a file to my previous commit", "commit_amend"),
    ("modify the last commit", "commit_amend"),
    ("amend commit with staged changes", "commit_amend"),
    ("fix my commit message", "commit_amend"),
    ("i want to update the last commit i made", "commit_amend"),
    ("change what was committed last time", "commit_amend"),
    ("append my staged files to the last commit", "commit_amend"),
    ("i forgot something in my last commit", "commit_amend"),

    # ── push ──────────────────────────────────────────────────────────────────
    ("push my changes to github", "push"),
    ("upload my commits to the remote", "push"),
    ("send my code to origin", "push"),
    ("push to remote", "push"),
    ("how do i push my code", "push"),
    ("i want to upload my work", "push"),
    ("push the current branch to remote", "push"),
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
    ("push commits to origin", "push"),

    # ── push_force ────────────────────────────────────────────────────────────
    ("force push my changes", "push_force"),
    ("git push --force", "push_force"),
    ("git push -f", "push_force"),
    ("i need to force push", "push_force"),
    ("overwrite the remote with my local branch", "push_force"),
    ("push force to origin", "push_force"),
    ("git push --force-with-lease", "push_force"),
    ("how do i force push safely", "push_force"),
    ("i rebased and need to force push", "push_force"),
    ("push even though the histories diverged", "push_force"),
    ("force update the remote branch", "push_force"),
    ("push with force flag", "push_force"),
    ("overwrite remote history with my local one", "push_force"),
    ("push force with lease to be safe", "push_force"),
    ("i need to push after a rebase", "push_force"),

    # ── push_upstream ─────────────────────────────────────────────────────────
    ("set upstream and push", "push_upstream"),
    ("git push -u origin", "push_upstream"),
    ("push and set the upstream tracking branch", "push_upstream"),
    ("push a new branch for the first time", "push_upstream"),
    ("how do i push a new branch to remote", "push_upstream"),
    ("set the tracking remote for my branch", "push_upstream"),
    ("push and track remote branch", "push_upstream"),
    ("git push --set-upstream origin feature", "push_upstream"),
    ("push my new branch and create the remote counterpart", "push_upstream"),
    ("push with upstream flag", "push_upstream"),
    ("create a remote tracking branch while pushing", "push_upstream"),
    ("first time pushing this branch", "push_upstream"),
    ("push and register the remote branch", "push_upstream"),
    ("link my local branch to a remote branch when pushing", "push_upstream"),
    ("how do i push a branch that doesnt exist on remote yet", "push_upstream"),

    # ── push_tags ─────────────────────────────────────────────────────────────
    ("push my tags to remote", "push_tags"),
    ("git push --tags", "push_tags"),
    ("how do i push tags", "push_tags"),
    ("push all tags to origin", "push_tags"),
    ("upload my version tags", "push_tags"),
    ("push git tags to github", "push_tags"),
    ("i created a tag, push it to remote", "push_tags"),
    ("how do i share my tags with the team", "push_tags"),
    ("push a specific tag to origin", "push_tags"),
    ("send my release tags to github", "push_tags"),
    ("push tags along with commits", "push_tags"),
    ("how do i push annotated tags", "push_tags"),
    ("upload tags to remote repository", "push_tags"),
    ("push version tags to bitbucket", "push_tags"),

    # ── pull ──────────────────────────────────────────────────────────────────
    ("get the latest code from remote", "pull"),
    ("pull changes from github", "pull"),
    ("download the latest commits", "pull"),
    ("sync my local repo with remote", "pull"),
    ("how do i get updates from the server", "pull"),
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
    ("i want to sync with the team changes", "pull"),
    ("sync my repo with the remote", "pull"),
    ("sync my local code with github", "pull"),
    ("get all the latest updates from origin", "pull"),

    # ── pull_rebase ───────────────────────────────────────────────────────────
    ("pull with rebase", "pull_rebase"),
    ("git pull --rebase", "pull_rebase"),
    ("fetch and rebase instead of merge", "pull_rebase"),
    ("pull using rebase strategy", "pull_rebase"),
    ("how do i pull without a merge commit", "pull_rebase"),
    ("update my branch using rebase not merge", "pull_rebase"),
    ("i want to pull but keep history linear", "pull_rebase"),
    ("pull rebase to avoid merge commits", "pull_rebase"),
    ("get remote changes and rebase my work on top", "pull_rebase"),
    ("pull with rebase flag", "pull_rebase"),
    ("sync using pull rebase", "pull_rebase"),
    ("pull --rebase origin main", "pull_rebase"),
    ("how do i pull with a clean linear history", "pull_rebase"),
    ("use rebase when pulling from remote", "pull_rebase"),

    # ── fetch ─────────────────────────────────────────────────────────────────
    ("fetch changes from remote without merging", "fetch"),
    ("git fetch", "fetch"),
    ("download remote changes but dont merge", "fetch"),
    ("i want to see what changed on remote without applying it", "fetch"),
    ("fetch origin", "fetch"),
    ("fetch all remotes", "fetch"),
    ("download the remote refs", "fetch"),
    ("how do i fetch without merging", "fetch"),
    ("get remote updates without applying them", "fetch"),
    ("fetch the latest from remote but keep my code as is", "fetch"),
    ("check what changed remotely without pulling", "fetch"),
    ("preview remote changes before pulling", "fetch"),
    ("update my remote tracking branches", "fetch"),
    ("fetch origin main", "fetch"),
    ("download remote commits to review later", "fetch"),
    ("how do i fetch all branches from remote", "fetch"),
    ("just fetch dont merge yet", "fetch"),
    ("get the remote metadata", "fetch"),
    ("fetch and see what others pushed", "fetch"),
    ("fetch remote branches without touching my work", "fetch"),

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCHING
    # ══════════════════════════════════════════════════════════════════════════

    # ── list_branches ─────────────────────────────────────────────────────────
    ("show me all the branches", "list_branches"),
    ("list all branches", "list_branches"),
    ("what branches exist", "list_branches"),
    ("which branches do i have", "list_branches"),
    ("show all branches in this repo", "list_branches"),
    ("how do i see all branches", "list_branches"),
    ("display all branches", "list_branches"),
    ("what are the available branches", "list_branches"),
    ("list every branch", "list_branches"),
    ("show me the branches", "list_branches"),
    ("can i see all the branches", "list_branches"),
    ("what branches are there", "list_branches"),
    ("give me a list of all branches", "list_branches"),
    ("show remote and local branches", "list_branches"),
    ("which branch am i on", "list_branches"),
    ("what is my current branch", "list_branches"),
    ("show me the current branch", "list_branches"),
    ("how do i check what branch i am on", "list_branches"),
    ("list all local and remote branches", "list_branches"),
    ("i want to see all the branches in my project", "list_branches"),
    ("show me all remote branches too", "list_branches"),
    ("git branch -a", "list_branches"),
    ("show including remote branches", "list_branches"),

    # ── create_branch ─────────────────────────────────────────────────────────
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
    ("can you create a branch for me and switch to it", "create_branch"),
    ("create a branch to work on a specific task", "create_branch"),
    ("make a branch that is separate from the main code", "create_branch"),
    ("how do i make a branch to work on a new feature", "create_branch"),
    ("create a branch to work on a bug fix", "create_branch"),
    ("new branch please", "create_branch"),
    ("spin up a branch called hotfix", "create_branch"),
    ("spin up a hotfix branch", "create_branch"),
    ("spin up a new branch for this task", "create_branch"),
    ("i need a feature branch", "create_branch"),
    ("make me a branch named dev", "create_branch"),
    ("create branch feature-login", "create_branch"),
    ("i want to isolate my changes in a new branch", "create_branch"),
    ("start a new branch for my fix", "create_branch"),
    ("git checkout -b new-branch", "create_branch"),
    ("git switch -c feature", "create_branch"),
    ("create and switch to a new branch", "create_branch"),

    # ── checkout ──────────────────────────────────────────────────────────────
    ("switch to main branch", "checkout"),
    ("go to the develop branch", "checkout"),
    ("checkout main", "checkout"),
    ("switch to master", "checkout"),
    ("how do i switch branches", "checkout"),
    ("i want to go back to main", "checkout"),
    ("move to a different branch", "checkout"),
    ("git checkout main", "checkout"),
    ("git switch develop", "checkout"),
    ("change to the feature branch", "checkout"),
    ("go to branch dev", "checkout"),
    ("switch over to the hotfix branch", "checkout"),
    ("i need to be on the main branch", "checkout"),
    ("check out the release branch", "checkout"),
    ("how do i move to another branch", "checkout"),
    ("navigate to the dev branch", "checkout"),
    ("jump to master", "checkout"),
    ("get on the main branch", "checkout"),
    ("i want to work on a different branch now", "checkout"),
    ("switch my working branch to main", "checkout"),

    # ── delete_branch ─────────────────────────────────────────────────────────
    ("delete a branch", "delete_branch"),
    ("remove a branch", "delete_branch"),
    ("i want to delete my feature branch", "delete_branch"),
    ("git branch -d feature", "delete_branch"),
    ("how do i delete a branch in git", "delete_branch"),
    ("clean up old branches", "delete_branch"),
    ("get rid of a branch i no longer need", "delete_branch"),
    ("delete the merged branch", "delete_branch"),
    ("force delete a branch", "delete_branch"),
    ("git branch -D old-feature", "delete_branch"),
    ("i merged my feature, now delete the branch", "delete_branch"),
    ("remove the hotfix branch", "delete_branch"),
    ("how do i clean up finished branches", "delete_branch"),
    ("delete this branch after merging", "delete_branch"),
    ("remove a stale branch", "delete_branch"),
    ("prune old branches", "delete_branch"),
    ("i dont need this branch anymore, delete it", "delete_branch"),
    ("how do i remove a local branch", "delete_branch"),
    ("wipe this branch", "delete_branch"),
    ("clean up the branch list", "delete_branch"),

    # ── merge_branch ──────────────────────────────────────────────────────────
    ("merge the changes from the feature branch into the main branch", "merge_branch"),
    ("can you merge the updates from the dev branch into the master branch", "merge_branch"),
    ("i want to merge the changes from the new-feature branch into main", "merge_branch"),
    ("how do i merge the fix branch into the main branch", "merge_branch"),
    ("can you merge the feature branch into the release branch", "merge_branch"),
    ("i need to merge the hotfix branch into the main branch", "merge_branch"),
    ("merge the updated branch into main", "merge_branch"),
    ("can you merge feature into develop", "merge_branch"),
    ("combine my branch with master", "merge_branch"),
    ("integrate my feature branch into main", "merge_branch"),
    ("bring the changes from dev into production", "merge_branch"),
    ("i want to merge my work back into the main branch", "merge_branch"),
    ("merge branches together", "merge_branch"),
    ("join my branch with the main codebase", "merge_branch"),
    ("how do i combine two branches", "merge_branch"),
    ("absorb the hotfix branch into main", "merge_branch"),
    ("i finished my feature, merge it in", "merge_branch"),
    ("fast-forward merge my branch", "merge_branch"),
    ("bring my branch into master", "merge_branch"),
    ("merge without a fast forward", "merge_branch"),

    # ── merge_squash ──────────────────────────────────────────────────────────
    ("squash and merge my feature", "merge_squash"),
    ("merge with squash", "merge_squash"),
    ("git merge --squash", "merge_squash"),
    ("squash all commits into one then merge", "merge_squash"),
    ("how do i squash commits when merging", "merge_squash"),
    ("merge squashing all my commits into one", "merge_squash"),
    ("combine all my commits into one before merging", "merge_squash"),
    ("i want a clean single commit when merging", "merge_squash"),
    ("squash merge the feature branch", "merge_squash"),
    ("merge with a single squashed commit", "merge_squash"),
    ("how do i squash merge into main", "merge_squash"),
    ("squash the commits and merge", "merge_squash"),
    ("flatten commits then merge into main", "merge_squash"),
    ("merge squash my branch", "merge_squash"),

    # ── merge_abort ───────────────────────────────────────────────────────────
    ("abort the merge", "merge_abort"),
    ("git merge --abort", "merge_abort"),
    ("cancel the current merge", "merge_abort"),
    ("stop the merge i started", "merge_abort"),
    ("how do i undo a merge in progress", "merge_abort"),
    ("exit from a merge with conflicts", "merge_abort"),
    ("i got merge conflicts, how do i cancel", "merge_abort"),
    ("go back to before the merge", "merge_abort"),
    ("undo the merge operation", "merge_abort"),
    ("i dont want to finish this merge", "merge_abort"),
    ("back out of the current merge", "merge_abort"),
    ("how do i abort a conflicted merge", "merge_abort"),
    ("cancel merge and restore my branch", "merge_abort"),
    ("git merge abort", "merge_abort"),

    # ── rebase ────────────────────────────────────────────────────────────────
    ("rebase my branch onto main", "rebase"),
    ("git rebase main", "rebase"),
    ("how do i rebase", "rebase"),
    ("rebase my feature branch on top of master", "rebase"),
    ("i want to rebase instead of merge", "rebase"),
    ("apply my commits on top of main", "rebase"),
    ("rebase onto the latest main", "rebase"),
    ("how do i keep my branch up to date with rebase", "rebase"),
    ("update my branch by rebasing", "rebase"),
    ("how do i rebase onto develop", "rebase"),
    ("rebase my work on top of the latest changes", "rebase"),
    ("use rebase to integrate upstream changes", "rebase"),
    ("rebase to make a cleaner history", "rebase"),
    ("git rebase develop", "rebase"),
    ("rebase this branch onto master", "rebase"),

    # ── rebase_interactive ────────────────────────────────────────────────────
    ("rebase interactively", "rebase_interactive"),
    ("git rebase -i", "rebase_interactive"),
    ("interactive rebase", "rebase_interactive"),
    ("squash my commits using rebase", "rebase_interactive"),
    ("clean up my commit history with rebase", "rebase_interactive"),
    ("rebase to fix my commits before merging", "rebase_interactive"),
    ("interactive rebase to squash commits", "rebase_interactive"),
    ("i want a cleaner commit history before merging", "rebase_interactive"),
    ("squash the last 3 commits into one", "rebase_interactive"),
    ("how do i rewrite my commit history", "rebase_interactive"),
    ("edit my commit messages using rebase", "rebase_interactive"),
    ("combine multiple commits into one using rebase", "rebase_interactive"),
    ("reorder my commits", "rebase_interactive"),
    ("drop some of my old commits", "rebase_interactive"),
    ("squash all my commits into one", "rebase_interactive"),
    ("use interactive rebase to clean history", "rebase_interactive"),

    # ── rebase_abort ──────────────────────────────────────────────────────────
    ("abort the rebase", "rebase_abort"),
    ("git rebase --abort", "rebase_abort"),
    ("cancel the rebase", "rebase_abort"),
    ("stop the rebase i started", "rebase_abort"),
    ("how do i cancel a rebase", "rebase_abort"),
    ("exit the rebase with conflicts", "rebase_abort"),
    ("undo the current rebase", "rebase_abort"),
    ("go back to before the rebase", "rebase_abort"),
    ("i got conflicts during rebase, cancel it", "rebase_abort"),
    ("how do i bail out of a rebase", "rebase_abort"),
    ("i want to quit the rebase", "rebase_abort"),
    ("back out of the rebase operation", "rebase_abort"),
    ("cancel rebase and return to original branch state", "rebase_abort"),

    # ── rebase_continue ───────────────────────────────────────────────────────
    ("continue the rebase after fixing conflicts", "rebase_continue"),
    ("git rebase --continue", "rebase_continue"),
    ("i fixed the conflicts, continue rebasing", "rebase_continue"),
    ("how do i resume a paused rebase", "rebase_continue"),
    ("continue rebase after resolving", "rebase_continue"),
    ("proceed with the rebase", "rebase_continue"),
    ("rebase continue", "rebase_continue"),
    ("i resolved the conflict, keep going with rebase", "rebase_continue"),
    ("resume the rebase operation", "rebase_continue"),
    ("how do i finish the rebase", "rebase_continue"),
    ("tell git to continue after conflict resolution", "rebase_continue"),
    ("move to the next step in the rebase", "rebase_continue"),
    ("after resolving conflicts continue rebase", "rebase_continue"),

    # ══════════════════════════════════════════════════════════════════════════
    # INSPECTION
    # ══════════════════════════════════════════════════════════════════════════

    # ── log ───────────────────────────────────────────────────────────────────
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
    ("what has been done in this repo", "log"),
    ("see the timeline of changes", "log"),
    ("show changes made by the team", "log"),
    ("what all has happened till now in this repo", "log"),
    ("what all has happened", "log"),
    ("what all changes have happened", "log"),
    ("what happened in this project", "log"),
    ("show me everything that happened", "log"),
    ("what is the history", "log"),
    ("what is the commit hostory", "log"),
    ("history of commits", "log"),
    ("see the project history", "log"),
    ("show me what has been done", "log"),
    ("what has happened so far", "log"),
    ("give me the full history", "log"),

    # ── log_oneline ───────────────────────────────────────────────────────────
    ("git log --oneline", "log_oneline"),
    ("show me a compact commit history", "log_oneline"),
    ("one line per commit", "log_oneline"),
    ("show commits in one line format", "log_oneline"),
    ("compact log view", "log_oneline"),
    ("how do i see a short summary of commits", "log_oneline"),
    ("display log with one line per commit", "log_oneline"),
    ("show commit hashes and messages only", "log_oneline"),
    ("abbreviated commit log", "log_oneline"),
    ("brief commit history", "log_oneline"),
    ("condensed commit list", "log_oneline"),
    ("short log", "log_oneline"),
    ("show just the commit titles", "log_oneline"),
    ("minimal log output", "log_oneline"),

    # ── log_graph ─────────────────────────────────────────────────────────────
    ("show me a visual branch graph", "log_graph"),
    ("git log --graph", "log_graph"),
    ("log with branch visualization", "log_graph"),
    ("show the commit tree", "log_graph"),
    ("display branches as a tree", "log_graph"),
    ("visual history of branches", "log_graph"),
    ("how do i see branching history visually", "log_graph"),
    ("show git history as a graph", "log_graph"),
    ("log graph with decorations", "log_graph"),
    ("show me the branch topology", "log_graph"),
    ("commit history with branch lines", "log_graph"),
    ("draw the commit graph", "log_graph"),
    ("pretty graph of all branches", "log_graph"),
    ("i want to see merges and branches visually", "log_graph"),

    # ── log_file ──────────────────────────────────────────────────────────────
    ("show commits that changed a specific file", "log_file"),
    ("git log for one file", "log_file"),
    ("history of a specific file", "log_file"),
    ("who changed this file and when", "log_file"),
    ("how do i see the commit history of a file", "log_file"),
    ("show all commits that touched this file", "log_file"),
    ("log for readme.md only", "log_file"),
    ("changes to a particular file over time", "log_file"),
    ("track the history of one file", "log_file"),
    ("which commits modified this file", "log_file"),
    ("show me when this file was changed", "log_file"),
    ("file-specific commit history", "log_file"),
    ("what happened to this file in git", "log_file"),
    ("see all edits to a particular file", "log_file"),

    # ── log_stat ──────────────────────────────────────────────────────────────
    ("show commit history with stats", "log_stat"),
    ("git log --stat", "log_stat"),
    ("show how many lines changed per commit", "log_stat"),
    ("log with file change statistics", "log_stat"),
    ("show insertions and deletions per commit", "log_stat"),
    ("commit history with number of changes", "log_stat"),
    ("how do i see how much changed in each commit", "log_stat"),
    ("log with file summaries", "log_stat"),
    ("display commit stats", "log_stat"),
    ("show lines added and removed per commit", "log_stat"),
    ("log showing changed files and line counts", "log_stat"),
    ("commits with a diff stat summary", "log_stat"),
    ("how many lines were changed in each commit", "log_stat"),
    ("show me the size of each commit", "log_stat"),

    # ── diff ──────────────────────────────────────────────────────────────────
    ("show me the diff", "diff"),
    ("what changed in my files", "diff"),
    ("show differences in my code", "diff"),
    ("how do i see what i changed before committing", "diff"),
    ("git diff", "diff"),
    ("show me line by line what changed", "diff"),
    ("what is different from the last version", "diff"),
    ("i want to review my changes", "diff"),
    ("see the changes i made", "diff"),
    ("show me unstaged changes", "diff"),
    ("what lines did i add or remove", "diff"),
    ("show file differences", "diff"),
    ("i want to see before and after", "diff"),
    ("preview my unstaged changes", "diff"),
    ("what exactly did i modify", "diff"),
    ("what lines did i change", "diff"),
    ("which lines were added or removed", "diff"),
    ("compare my working directory to last commit", "diff"),
    ("show changes in my tracked files", "diff"),

    # ── diff_staged ───────────────────────────────────────────────────────────
    ("show me what is staged for commit", "diff_staged"),
    ("diff staged changes", "diff_staged"),
    ("git diff --staged", "diff_staged"),
    ("git diff --cached", "diff_staged"),
    ("what will be in my next commit", "diff_staged"),
    ("show me staged vs last commit", "diff_staged"),
    ("compare what i staged to the last commit", "diff_staged"),
    ("see the staged diff", "diff_staged"),
    ("what changes are ready to commit", "diff_staged"),
    ("how do i see what i have staged", "diff_staged"),
    ("show me the diff of staged files", "diff_staged"),
    ("what exactly will i commit", "diff_staged"),
    ("review staged changes before committing", "diff_staged"),
    ("see what is in the staging area compared to last commit", "diff_staged"),
    ("diff cached", "diff_staged"),
    ("show staged modifications", "diff_staged"),
    ("what has been added to the index", "diff_staged"),
    ("preview what will be committed", "diff_staged"),

    # ── diff_branches ─────────────────────────────────────────────────────────
    ("compare two branches", "diff_branches"),
    ("diff between main and feature", "diff_branches"),
    ("show differences between two branches", "diff_branches"),
    ("git diff main feature", "diff_branches"),
    ("what is different between my branch and main", "diff_branches"),
    ("how do i compare two branches", "diff_branches"),
    ("see what changed between branches", "diff_branches"),
    ("show the diff of two branches", "diff_branches"),
    ("compare feature branch to main", "diff_branches"),
    ("what commits are in feature but not in main", "diff_branches"),
    ("how does my branch differ from master", "diff_branches"),
    ("diff my branch against main", "diff_branches"),
    ("compare develop and main", "diff_branches"),
    ("see what my branch adds compared to main", "diff_branches"),
    ("branch diff", "diff_branches"),

    # ── diff_commit ───────────────────────────────────────────────────────────
    ("compare to a specific commit", "diff_commit"),
    ("git diff HEAD~1", "diff_commit"),
    ("show changes since a particular commit", "diff_commit"),
    ("diff against a commit hash", "diff_commit"),
    ("how do i compare my current state to an older commit", "diff_commit"),
    ("what changed since commit abc123", "diff_commit"),
    ("show me what changed between two commits", "diff_commit"),
    ("compare head to two commits ago", "diff_commit"),
    ("diff current with previous commit", "diff_commit"),
    ("see what changed since last release tag", "diff_commit"),
    ("compare my current code to an older version", "diff_commit"),
    ("diff from HEAD~2 to now", "diff_commit"),
    ("show what changed between two commit hashes", "diff_commit"),
    ("how do i see changes between commits", "diff_commit"),

    # ── show ──────────────────────────────────────────────────────────────────
    ("show me what was in a specific commit", "show"),
    ("git show", "show"),
    ("what did the last commit contain", "show"),
    ("show me the details of a commit", "show"),
    ("inspect a commit", "show"),
    ("what changed in commit abc123", "show"),
    ("show the content of that commit", "show"),
    ("how do i see what a commit did", "show"),
    ("show commit details", "show"),
    ("show me the full diff of a commit", "show"),
    ("display a specific commit", "show"),
    ("what was in the previous commit", "show"),
    ("show the changes from the last commit", "show"),
    ("inspect what was committed", "show"),
    ("show me the patch for a commit", "show"),
    ("view a specific commit", "show"),
    ("git show HEAD", "show"),
    ("what changes did that commit introduce", "show"),
    ("see what a specific commit changed", "show"),
    ("show me the commit content", "show"),

    # ── blame ─────────────────────────────────────────────────────────────────
    ("who wrote this line of code", "blame"),
    ("git blame", "blame"),
    ("who last changed this file", "blame"),
    ("show me who edited each line", "blame"),
    ("blame this file", "blame"),
    ("find out who wrote a particular line", "blame"),
    ("how do i see who changed each line", "blame"),
    ("annotate the file with author info", "blame"),
    ("who is responsible for this code", "blame"),
    ("show per line author history", "blame"),
    ("git blame on this file", "blame"),
    ("find the author of a specific line", "blame"),
    ("who committed this line", "blame"),
    ("show me line by line who wrote this", "blame"),
    ("i want to know who made these changes", "blame"),
    ("track down who wrote this function", "blame"),
    ("show commit and author per line", "blame"),
    ("annotate source with blame info", "blame"),

    # ══════════════════════════════════════════════════════════════════════════
    # UNDOING
    # ══════════════════════════════════════════════════════════════════════════

    # ── restore ───────────────────────────────────────────────────────────────
    ("restore a file to its last committed state", "restore"),
    ("git restore", "restore"),
    ("discard changes in a single file", "restore"),
    ("undo changes to one specific file", "restore"),
    ("revert a file back to what it was", "restore"),
    ("i edited a file by mistake, undo just that file", "restore"),
    ("how do i discard changes in one file", "restore"),
    ("put this file back to how it was", "restore"),
    ("restore this file to head", "restore"),
    ("throw away my edits to a specific file", "restore"),
    ("discard edits in readme.md", "restore"),
    ("clean up just this one file", "restore"),
    ("how do i undo changes to a file without affecting others", "restore"),
    ("reset one file to last commit", "restore"),
    ("bring a file back to its committed version", "restore"),
    ("i messed up one file, restore it", "restore"),
    ("revert a specific file", "restore"),
    ("undo changes to a file", "restore"),
    ("put a file back to its original state", "restore"),

    # ── revert ────────────────────────────────────────────────────────────────
    ("revert the last commit", "revert"),
    ("git revert HEAD", "revert"),
    ("undo a commit by making a new commit", "revert"),
    ("safely undo a commit", "revert"),
    ("how do i undo a pushed commit safely", "revert"),
    ("create a reverting commit", "revert"),
    ("i want to undo a commit but keep the history", "revert"),
    ("revert a specific commit", "revert"),
    ("undo without rewriting history", "revert"),
    ("how do i reverse a commit safely", "revert"),
    ("make a commit that undoes the last one", "revert"),
    ("i already pushed, how do i undo", "revert"),
    ("revert head to undo", "revert"),
    ("create an undo commit", "revert"),
    ("how do i safely revert a merged commit", "revert"),
    ("roll back a commit with a new commit", "revert"),
    ("i need to undo a commit that was already pushed", "revert"),
    ("git revert to undo", "revert"),
    ("make a commit that reverses changes", "revert"),

    # ── reset ─────────────────────────────────────────────────────────────────
    ("undo my last commit", "reset"),
    ("how do i undo in git", "reset"),
    ("i made a mistake, undo it", "reset"),
    ("remove my last commit", "reset"),
    ("i accidentally committed, undo it", "reset"),
    ("how do i go back to a previous state", "reset"),
    ("i want to undo a commit", "reset"),
    ("undo staged changes", "reset"),
    ("unstage my files", "reset"),
    ("git reset", "reset"),
    ("reset my repo", "reset"),
    ("take back my last commit", "reset"),

    # ── reset_soft ────────────────────────────────────────────────────────────
    ("soft reset my last commit", "reset_soft"),
    ("git reset --soft HEAD~1", "reset_soft"),
    ("undo the last commit but keep my changes staged", "reset_soft"),
    ("how do i uncommit but keep my work ready to commit again", "reset_soft"),
    ("undo the commit and leave changes staged", "reset_soft"),
    ("i want to undo the last commit but not lose any work", "reset_soft"),
    ("take back the commit but keep everything staged", "reset_soft"),
    ("remove the last commit but keep the index intact", "reset_soft"),
    ("uncommit but preserve staged files", "reset_soft"),
    ("how do i undo a commit softly", "reset_soft"),
    ("go back one commit and keep staged changes", "reset_soft"),
    ("soft reset to undo commit only", "reset_soft"),
    ("undo commit but keep changes in staging area", "reset_soft"),
    ("reset HEAD but keep staged", "reset_soft"),

    # ── reset_hard ────────────────────────────────────────────────────────────
    ("hard reset my branch", "reset_hard"),
    ("git reset --hard HEAD~1", "reset_hard"),
    ("discard all changes and go back one commit", "reset_hard"),
    ("nuke everything and reset to last commit", "reset_hard"),
    ("throw away all my local changes completely", "reset_hard"),
    ("hard reset to the previous commit", "reset_hard"),
    ("i want to wipe out all my changes", "reset_hard"),
    ("reset and lose all my uncommitted work", "reset_hard"),
    ("discard working directory and staging area", "reset_hard"),
    ("how do i hard reset in git", "reset_hard"),
    ("go back to a previous commit and delete everything after it", "reset_hard"),
    ("force reset to a specific commit", "reset_hard"),
    ("reset --hard to clean slate", "reset_hard"),
    ("wipe all local changes with a hard reset", "reset_hard"),
    ("hard reset and start fresh from the last commit", "reset_hard"),

    # ── reset_mixed ───────────────────────────────────────────────────────────
    ("mixed reset my last commit", "reset_mixed"),
    ("git reset HEAD~1", "reset_mixed"),
    ("undo the commit and unstage the changes", "reset_mixed"),
    ("how do i uncommit and unstage at the same time", "reset_mixed"),
    ("undo commit and unstage but keep files", "reset_mixed"),
    ("default reset behaviour", "reset_mixed"),
    ("go back one commit and unstage changes", "reset_mixed"),
    ("reset keeping my working directory intact", "reset_mixed"),
    ("how do i undo a commit but keep my edits", "reset_mixed"),
    ("unstage and uncommit at the same time", "reset_mixed"),
    ("reset to HEAD and unstage", "reset_mixed"),
    ("undo commit and remove from staging but keep disk changes", "reset_mixed"),
    ("soft reset but also unstage", "reset_mixed"),

    # ── reset_unstage ─────────────────────────────────────────────────────────
    ("unstage a specific file", "reset_unstage"),
    ("git reset HEAD <file>", "reset_unstage"),
    ("remove a file from the staging area", "reset_unstage"),
    ("how do i unstage a file", "reset_unstage"),
    ("undo git add for one file", "reset_unstage"),
    ("take a file out of the staging area", "reset_unstage"),
    ("i accidentally staged a file, unstage it", "reset_unstage"),
    ("unstage readme.md", "reset_unstage"),
    ("remove from staging without discarding changes", "reset_unstage"),
    ("how do i undo a git add", "reset_unstage"),
    ("unstage a file but keep my edits", "reset_unstage"),
    ("pull a file back out of the index", "reset_unstage"),
    ("undo staging for one file", "reset_unstage"),
    ("how do i remove a file from staging", "reset_unstage"),

    # ── stash ─────────────────────────────────────────────────────────────────
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
    ("push my changes onto the stash", "stash"),
    ("put changes in the stash stack", "stash"),

    # ── stash_pop ─────────────────────────────────────────────────────────────
    ("pop the stash", "stash_pop"),
    ("git stash pop", "stash_pop"),
    ("restore my stashed work", "stash_pop"),
    ("get my stashed changes back", "stash_pop"),
    ("bring back my saved changes from the stash", "stash_pop"),
    ("how do i apply a stash", "stash_pop"),
    ("unstash my changes", "stash_pop"),
    ("retrieve stashed work", "stash_pop"),
    ("i want to apply what i stashed", "stash_pop"),
    ("re-apply my stash", "stash_pop"),
    ("resume my stashed work", "stash_pop"),
    ("apply last stash entry", "stash_pop"),
    ("pop my saved changes from the stash", "stash_pop"),
    ("restore changes i set aside earlier", "stash_pop"),
    ("bring back the hidden changes", "stash_pop"),
    ("i want to continue my stashed work", "stash_pop"),
    ("apply the stash and remove it from stack", "stash_pop"),

    # ── stash_apply ───────────────────────────────────────────────────────────
    ("apply a specific stash entry", "stash_apply"),
    ("git stash apply stash@{1}", "stash_apply"),
    ("apply stash without removing it from the list", "stash_apply"),
    ("how do i apply an older stash", "stash_apply"),
    ("apply stash number 2", "stash_apply"),
    ("use a specific stash but keep it in the list", "stash_apply"),
    ("apply the second stash entry", "stash_apply"),
    ("how do i apply a stash and keep it", "stash_apply"),
    ("apply stash at index 1", "stash_apply"),
    ("restore a stash without dropping it", "stash_apply"),
    ("apply a named stash", "stash_apply"),
    ("reapply stash but keep it for later", "stash_apply"),
    ("git stash apply to use without popping", "stash_apply"),

    # ── stash_list ────────────────────────────────────────────────────────────
    ("list all stashes", "stash_list"),
    ("git stash list", "stash_list"),
    ("show all my saved stashes", "stash_list"),
    ("how do i see all my stashed work", "stash_list"),
    ("what stashes do i have", "stash_list"),
    ("show all stash entries", "stash_list"),
    ("display the stash stack", "stash_list"),
    ("how many stashes have i saved", "stash_list"),
    ("view stash history", "stash_list"),
    ("show me a list of all my stashes", "stash_list"),
    ("what is in my stash", "stash_list"),
    ("list the stash stack", "stash_list"),
    ("show all temporarily saved work", "stash_list"),

    # ── stash_drop ────────────────────────────────────────────────────────────
    ("delete a stash entry", "stash_drop"),
    ("git stash drop", "stash_drop"),
    ("remove a stash i no longer need", "stash_drop"),
    ("drop the stash", "stash_drop"),
    ("how do i delete a saved stash", "stash_drop"),
    ("clear a specific stash entry", "stash_drop"),
    ("discard a stash i dont want anymore", "stash_drop"),
    ("get rid of an old stash", "stash_drop"),
    ("remove stash at index 0", "stash_drop"),
    ("delete the last stash", "stash_drop"),
    ("i dont need this stash anymore, delete it", "stash_drop"),
    ("remove old stash entries", "stash_drop"),
    ("how do i clean up my stashes", "stash_drop"),

    # ══════════════════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════

    # ── remove_file ───────────────────────────────────────────────────────────
    ("remove a file from git", "remove_file"),
    ("delete a file and stop tracking it", "remove_file"),
    ("git rm this file", "remove_file"),
    ("how do i remove a file from the repo", "remove_file"),
    ("untrack a file and delete it", "remove_file"),
    ("delete a file from git history", "remove_file"),
    ("remove a tracked file", "remove_file"),
    ("stop tracking and delete this file", "remove_file"),
    ("git rm readme.md", "remove_file"),
    ("how do i delete a file using git", "remove_file"),
    ("remove a file from the repository", "remove_file"),
    ("how do i remove a committed file", "remove_file"),
    ("git remove file from index", "remove_file"),
    ("delete the file from the project and git", "remove_file"),
    ("remove file from staging and disk", "remove_file"),
    ("i want to delete a file and commit it", "remove_file"),

    # ── move_file ─────────────────────────────────────────────────────────────
    ("rename a file in git", "move_file"),
    ("move a file to another folder", "move_file"),
    ("git mv old new", "move_file"),
    ("how do i rename a file with git", "move_file"),
    ("i want to move a file and keep git history", "move_file"),
    ("rename this file and track the change", "move_file"),
    ("move file to a different directory", "move_file"),
    ("git move file", "move_file"),
    ("how do i rename a tracked file", "move_file"),
    ("rename index.html to home.html in git", "move_file"),
    ("move a file while preserving history", "move_file"),
    ("git rename file", "move_file"),
    ("change the name of a tracked file", "move_file"),
    ("reorganize files while keeping git history", "move_file"),
    ("move a file in my repo", "move_file"),
    ("how do i move files in git", "move_file"),
    ("rename a file so git knows about it", "move_file"),
]

# ---------------------------------------------------------------------------
# All recognised intents  (50 total)
# ---------------------------------------------------------------------------
ALL_INTENTS: list[str] = [
    # Setup & config
    "init", "clone", "clone_branch", "clone_shallow",
    "remote", "remote_add", "remote_remove",
    # Core workflow
    "status", "add", "add_patch",
    "commit", "commit_amend",
    "push", "push_force", "push_upstream", "push_tags",
    "pull", "pull_rebase", "fetch",
    # Branching
    "list_branches", "create_branch", "checkout", "delete_branch",
    "merge_branch", "merge_squash", "merge_abort",
    "rebase", "rebase_interactive", "rebase_abort", "rebase_continue",
    # Inspection
    "log", "log_oneline", "log_graph", "log_file", "log_stat",
    "diff", "diff_staged", "diff_branches", "diff_commit",
    "show", "blame",
    # Undoing
    "restore", "revert",
    "reset", "reset_soft", "reset_hard", "reset_mixed", "reset_unstage",
    "stash", "stash_pop", "stash_apply", "stash_list", "stash_drop",
    # File ops
    "remove_file", "move_file",
]

# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.6

# ---------------------------------------------------------------------------
# Keyword fallback rules
#
# ORDERING IS CRITICAL:
#   - Specific variants BEFORE their parent (reset_soft before reset,
#     stash_pop before stash, log_oneline before log, etc.)
#   - Longer phrases within a rule are tried before shorter ones (sorted
#     at runtime by length descending).
# ---------------------------------------------------------------------------
_KEYWORD_RULES: list[tuple[list[str], str]] = [

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
    (["switch to main", "switch to master", "switch to develop",
      "go to branch", "checkout main", "checkout master",
      "git switch ", "git checkout ", "change to the branch",
      "move to another branch", "navigate to branch",
      "get on the main branch", "jump to master",
      "go to the branch", "i want to be on branch"], "checkout"),

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
      "what has been changed"], "status"),

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
      "record my changes", "store my work"], "commit"),

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

# ---------------------------------------------------------------------------
# Internal model handle
# ---------------------------------------------------------------------------
_pipeline: Optional[object] = None


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"['\"]", "", text)
    text = re.sub(r"[^a-z0-9\s\-\.]", " ", text)
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
            ngram_range=(1, 3),     # unigrams + bigrams + trigrams for variants
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def _keyword_fallback(text: str) -> Optional[str]:
    """
    Return intent matched by keyword rules, or None.
    Rules evaluated top-to-bottom (declaration order); first match wins.
    Within each rule's keyword list longer phrases are tried before shorter
    ones to prevent premature short-circuit matches.
    """
    lower = text.lower()
    for keywords, intent in _KEYWORD_RULES:
        for kw in sorted(keywords, key=len, reverse=True):
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
    >>> classify("undo my last commit but keep changes staged")
    ('reset_soft', 0.84)
    >>> classify("squash the last 3 commits into one")
    ('rebase_interactive', 0.79)
    >>> classify("show me a branch graph")
    ('log_graph', 0.88)
    """
    global _pipeline

    text = (user_input or "").strip()
    if not text:
        return "unknown", 0.0

    # ── ML classification ────────────────────────────────────────────────────
    if _pipeline is None:
        _pipeline = _build_pipeline()

    normalised  = _normalise(text)
    proba_array = _pipeline.predict_proba([normalised])[0]
    best_idx    = int(proba_array.argmax())
    confidence  = float(proba_array[best_idx])
    ml_intent   = _pipeline.classes_[best_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        return ml_intent, round(confidence, 4)

    # ── keyword fallback ─────────────────────────────────────────────────────
    fallback = _keyword_fallback(text)
    if fallback:
        return fallback, 1.0

    return "unknown", 0.0


def detect_intent(user_input: str) -> str:
    """
    Legacy single-value wrapper for backwards compatibility.
    Prefer classify() for new code.
    """
    intent, _ = classify(user_input)
    return intent


# ---------------------------------------------------------------------------
# Warm up on import
# ---------------------------------------------------------------------------
def _warm_up() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()


_warm_up()


# ---------------------------------------------------------------------------
# Self-test:  python intent.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _TEST_CASES: list[tuple[str, str]] = [
        # ── real failures from previous sessions ──────────────────────────
        ("show me all the branches",                              "list_branches"),
        ("what all has happened till now in this repo",           "log"),
        ("what is the commit hostory",                            "log"),

        # ── init / clone variants ─────────────────────────────────────────
        ("initialize git in my folder",                           "init"),
        ("clone this github repo",                                "clone"),
        ("clone only the develop branch",                         "clone_branch"),
        ("clone without the full history",                        "clone_shallow"),
        ("git clone --depth 1",                                   "clone_shallow"),

        # ── remote variants ───────────────────────────────────────────────
        ("show me my remotes",                                    "remote"),
        ("add a remote origin",                                   "remote_add"),
        ("git remote add origin",                                 "remote_add"),
        ("remove the origin remote",                              "remote_remove"),

        # ── status / add / add_patch ──────────────────────────────────────
        ("which files are modified",                              "status"),
        ("stage all my changes",                                  "add"),
        ("git add -p",                                            "add_patch"),
        ("let me pick which changes to stage",                    "add_patch"),

        # ── commit / commit_amend ─────────────────────────────────────────
        ("commit all my changes with a message",                  "commit"),
        ("amend my last commit",                                  "commit_amend"),
        ("i forgot to include a file in my last commit",          "commit_amend"),
        ("fix the commit message",                                "commit_amend"),

        # ── push variants ─────────────────────────────────────────────────
        ("push to github",                                        "push"),
        ("force push after rebase",                               "push_force"),
        ("git push --force",                                      "push_force"),
        ("push and set upstream tracking",                        "push_upstream"),
        ("git push -u origin main",                               "push_upstream"),
        ("push my version tags",                                  "push_tags"),
        ("git push --tags",                                       "push_tags"),

        # ── pull / pull_rebase / fetch ────────────────────────────────────
        ("get the latest code from remote",                       "pull"),
        ("pull with rebase",                                      "pull_rebase"),
        ("git pull --rebase",                                     "pull_rebase"),
        ("fetch changes without merging",                         "fetch"),
        ("just fetch dont merge yet",                             "fetch"),

        # ── branch operations ─────────────────────────────────────────────
        ("list all branches",                                     "list_branches"),
        ("git branch -a",                                         "list_branches"),
        ("create a new branch for my feature",                    "create_branch"),
        ("git checkout -b feature",                               "create_branch"),
        ("git switch -c feature",                                 "create_branch"),
        ("switch to main branch",                                 "checkout"),
        ("go to the develop branch",                              "checkout"),
        ("delete the feature branch",                             "delete_branch"),
        ("i dont need this branch anymore",                       "delete_branch"),

        # ── merge variants ────────────────────────────────────────────────
        ("merge my feature into master",                          "merge_branch"),
        ("squash and merge the feature branch",                   "merge_squash"),
        ("git merge --squash",                                    "merge_squash"),
        ("abort the merge",                                       "merge_abort"),
        ("git merge --abort",                                     "merge_abort"),

        # ── rebase variants ───────────────────────────────────────────────
        ("rebase my branch onto main",                            "rebase"),
        ("interactive rebase",                                    "rebase_interactive"),
        ("squash the last 3 commits into one",                    "rebase_interactive"),
        ("git rebase -i HEAD~3",                                  "rebase_interactive"),
        ("abort the rebase",                                      "rebase_abort"),
        ("git rebase --abort",                                    "rebase_abort"),
        ("continue the rebase after fixing conflicts",            "rebase_continue"),
        ("git rebase --continue",                                 "rebase_continue"),

        # ── log variants ──────────────────────────────────────────────────
        ("show me the commit history",                            "log"),
        ("git log --oneline",                                     "log_oneline"),
        ("show me a compact commit history",                      "log_oneline"),
        ("show me a visual branch graph",                         "log_graph"),
        ("git log --graph",                                       "log_graph"),
        ("show commits that changed a specific file",             "log_file"),
        ("history of a specific file",                            "log_file"),
        ("show commit history with stats",                        "log_stat"),
        ("git log --stat",                                        "log_stat"),

        # ── diff variants ─────────────────────────────────────────────────
        ("show me the diff",                                      "diff"),
        ("show me what is staged for commit",                     "diff_staged"),
        ("git diff --staged",                                     "diff_staged"),
        ("git diff --cached",                                     "diff_staged"),
        ("compare two branches",                                  "diff_branches"),
        ("diff between main and feature",                         "diff_branches"),
        ("compare to a specific commit",                          "diff_commit"),
        ("git diff HEAD~1",                                       "diff_commit"),

        # ── show / blame ──────────────────────────────────────────────────
        ("inspect a specific commit",                             "show"),
        ("git show HEAD",                                         "show"),
        ("who wrote this line of code",                           "blame"),
        ("git blame on this file",                                "blame"),

        # ── reset variants ────────────────────────────────────────────────
        ("undo my last commit",                                   "reset"),
        ("git reset --soft HEAD~1",                               "reset_soft"),
        ("undo commit but keep changes staged",                   "reset_soft"),
        ("keep changes staged after undoing commit",              "reset_soft"),
        ("git reset --hard HEAD~1",                               "reset_hard"),
        ("nuke everything and reset",                             "reset_hard"),
        ("discard all changes and go back one commit",            "reset_hard"),
        ("undo commit and unstage everything",                    "reset_mixed"),
        ("go back one commit and unstage",                        "reset_mixed"),
        ("unstage a specific file",                               "reset_unstage"),
        ("undo git add for one file",                             "reset_unstage"),
        ("remove a file from the staging area",                   "reset_unstage"),

        # ── restore / revert ──────────────────────────────────────────────
        ("restore a file to its last committed state",            "restore"),
        ("undo changes to one file",                              "restore"),
        ("revert the last commit",                                "revert"),
        ("git revert HEAD",                                       "revert"),
        ("i already pushed, how do i undo",                       "revert"),

        # ── stash variants ────────────────────────────────────────────────
        ("stash my changes",                                      "stash"),
        ("git stash",                                             "stash"),
        ("pop the stash",                                         "stash_pop"),
        ("git stash pop",                                         "stash_pop"),
        ("apply a specific stash entry",                          "stash_apply"),
        ("git stash apply stash@{1}",                             "stash_apply"),
        ("list all stashes",                                      "stash_list"),
        ("git stash list",                                        "stash_list"),
        ("delete a stash entry",                                  "stash_drop"),
        ("git stash drop",                                        "stash_drop"),

        # ── file ops ──────────────────────────────────────────────────────
        ("remove a file from git",                                "remove_file"),
        ("git rm this file",                                      "remove_file"),
        ("rename a file in git",                                  "move_file"),
        ("git mv old.txt new.txt",                                "move_file"),
    ]

    from collections import Counter
    print("\n── Dataset stats ─────────────────────────────────────────────")
    dist = Counter(label for _, label in _TRAINING_DATA)
    for intent in sorted(ALL_INTENTS):
        count = dist.get(intent, 0)
        bar = "█" * count
        print(f"  {intent:<22} {count:>3}  {bar}")
    print(f"\n  Total samples : {len(_TRAINING_DATA)}")
    print(f"  Total intents : {len(dist)}\n")

    print(f"{'INPUT':<57} {'EXPECTED':<22} {'GOT':<22} {'CONF':>6}  OK?")
    print("─" * 116)
    passed = 0
    for phrase, expected in _TEST_CASES:
        got, conf = classify(phrase)
        ok = "✓" if got == expected else "✗"
        if got == expected:
            passed += 1
        print(f"{phrase:<57} {expected:<22} {got:<22} {conf:>6.2f}  {ok}")

    total = len(_TEST_CASES)
    print(f"\n{'─' * 116}")
    print(f"Result: {passed}/{total} passed  ({100 * passed // total}%)\n")

