#!/usr/bin/env python3
"""
Upload (push) this project to GitHub: https://github.com/akashdutta1030hr-beep/Dyck_Task

Usage:
    python upload.py                    # commit all changes, push to main
    python upload.py "your commit msg"  # custom commit message

Requires: git installed, GitHub auth (HTTPS token or SSH key).
"""

import subprocess
import sys
import os

GITHUB_REPO = "https://github.com/akashdutta1030hr-beep/Dyck_Task"
REMOTE_NAME = "origin"
BRANCH = "main"


def run(cmd, check=True, capture=False):
    """Run a shell command."""
    kwargs = {"shell": True, "cwd": os.path.dirname(os.path.abspath(__file__))}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    r = subprocess.run(cmd, **kwargs)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(root, ".git")):
        print("Not a git repo. Run: git init")
        run("git init")
        run(f"git remote add {REMOTE_NAME} {GITHUB_REPO}")
        print(f"Initialized git and set remote to {GITHUB_REPO}")

    # Ensure remote points to user's repo
    try:
        out = run(f"git remote get-url {REMOTE_NAME}", capture=True)
        url = (out.stdout or out.stderr or "").strip()
        if GITHUB_REPO not in url and "akashdutta1030hr-beep/Dyck_Task" not in url:
            run(f"git remote set-url {REMOTE_NAME} {GITHUB_REPO}")
            print(f"Set remote to {GITHUB_REPO}")
    except subprocess.CalledProcessError:
        run(f"git remote add {REMOTE_NAME} {GITHUB_REPO}")
        print(f"Added remote {GITHUB_REPO}")

    run("git add -A")
    status = run("git status --short", capture=True)
    out = (status.stdout or status.stderr or "").strip()
    if not out:
        print("Nothing to commit (working tree clean). Pushing existing commits...")
        run(f"git push -u {REMOTE_NAME} {BRANCH}")
        return

    msg = (sys.argv[1] if len(sys.argv) > 1 else "Update Dyck Task").strip()
    msg = msg.replace('"', '\\"')  # escape for shell
    run(f'git commit -m "{msg}"')
    run(f"git push -u {REMOTE_NAME} {BRANCH}")
    print(f"Pushed to {GITHUB_REPO} (branch: {BRANCH})")


if __name__ == "__main__":
    main()
