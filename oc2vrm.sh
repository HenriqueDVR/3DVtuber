#!/bin/sh
# Starts the guided app (python -m app.server) on macOS and Linux: in the browser, or its own window
# with pywebview installed. On a new computer it first says what is missing, and asks before it
# installs the one package the app needs. Python itself comes from the system: this says how.
# Not yet run on a Mac or a Linux computer by the app's author: say what happens.
cd "$(dirname "$0")" || exit 1

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo
  echo " Python 3.10 or newer was not found. The app needs it to run."
  echo
  if [ "$(uname)" = "Darwin" ]; then
    echo " macOS: install it from https://www.python.org/downloads/ (its installer brings everything),"
    echo "        or with Homebrew:  brew install python@3.12 python-tk@3.12"
  else
    echo " Debian / Ubuntu:  sudo apt install python3 python3-pip python3-tk"
    echo " Fedora:           sudo dnf install python3 python3-pip python3-tkinter"
    echo " Arch:             sudo pacman -S python python-pip tk"
  fi
  echo
  echo " Then start this again."
  exit 1
fi

if ! "$PY" -c 'import PIL' >/dev/null 2>&1; then
  echo
  echo " First start: the app needs Python's picture library, Pillow (about 3 MB, from pypi.org)."
  printf " Install it now? [Y/n] "
  # nobody there to answer (started from a script): nothing is installed unasked
  read -r answer || answer=n
  case "$answer" in
    n|N|no|No) ;;
    *)
      # --user: beside the system's packages, not among them. A system that manages its own Python
      # refuses pip (PEP 668): then its package manager has Pillow.
      if ! "$PY" -m pip install --user -r requirements.txt; then
        echo
        echo " pip could not install it. Your system's own package has it:"
        echo "   Debian / Ubuntu:  sudo apt install python3-pil"
        echo "   Fedora:           sudo dnf install python3-pillow"
        echo "   Arch:             sudo pacman -S python-pillow"
        echo " The app starts anyway and says what is missing on its Get set up page."
      fi
      ;;
  esac
  echo
fi

if ! "$PY" -c 'import tkinter' >/dev/null 2>&1; then
  echo " Note: this Python has no tkinter, so the app's Browse buttons won't open (paths can be typed)."
  echo "       Debian / Ubuntu: sudo apt install python3-tk    macOS Homebrew: brew install python-tk"
  echo
fi

# The app asks to be started again (exit code 75) after it has updated itself. A new launcher is
# waiting as oc2vrm.sh.new then: a script can't be written over while it runs, so it is taken here.
export OC2VRM_LAUNCHER=1
while :; do
  "$PY" -m app.server "$@"
  code=$?
  [ "$code" -eq 75 ] || exit "$code"
  export OC2VRM_RESTARTED=1
  if [ -f "$0.new" ]; then mv "$0.new" "$0" && chmod +x "$0"; exec sh "$0" "$@"; fi
done
