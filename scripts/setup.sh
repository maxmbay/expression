#!/usr/bin/env bash
# Local setup: direnv, .env with HF_TOKEN, and Python dependencies. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v direnv >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing direnv with Homebrew..."
    brew install direnv
  else
    echo "direnv is not installed and Homebrew is unavailable; install direnv manually." >&2
    exit 1
  fi
fi

shell_name="$(basename "${SHELL:-zsh}")"
case "$shell_name" in
  zsh) rc_file="$HOME/.zshrc" ;;
  bash) rc_file="$HOME/.bashrc" ;;
  *) rc_file="" ;;
esac
if [[ -n "$rc_file" ]] && ! grep -q 'direnv hook' "$rc_file" 2>/dev/null; then
  printf '\neval "$(direnv hook %s)"\n' "$shell_name" >>"$rc_file"
  echo "Added the direnv hook to $rc_file (open a new shell to activate it)."
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env from .env.example."
fi

current_token="$(grep -E '^HF_TOKEN=' .env | head -n1 | cut -d= -f2- || true)"
if [[ -z "$current_token" ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Hugging Face token (input hidden, Enter to skip): " token
    echo
  else
    token="${HF_TOKEN:-}"
  fi
  if [[ -n "$token" ]]; then
    tmp="$(mktemp)"
    grep -v -E '^HF_TOKEN=' .env >"$tmp" || true
    printf 'HF_TOKEN=%s\n' "$token" >>"$tmp"
    mv "$tmp" .env
    chmod 600 .env
    echo "Saved HF_TOKEN to .env."
  else
    echo "HF_TOKEN left empty; edit .env later."
  fi
else
  echo "HF_TOKEN already set in .env."
fi

direnv allow .

if command -v uv >/dev/null 2>&1; then
  uv sync
else
  echo "uv not found; install it (brew install uv) and run 'uv sync'." >&2
fi

echo "Done. In this directory, HF_TOKEN loads automatically via direnv."
