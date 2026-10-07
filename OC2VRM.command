#!/bin/sh
# macOS: double-click this in Finder to start the app (it opens a Terminal window and runs oc2vrm.sh).
# The first time, macOS may ask to confirm: right-click the file, Open, Open.
exec "$(dirname "$0")/oc2vrm.sh" "$@"
