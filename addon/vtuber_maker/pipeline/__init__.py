"""Pipeline stages, each usable from the add-on UI or a headless Blender run."""

# every headless run imports the pipeline: an old Blender stops here with an error code the app knows
from . import compat  # noqa: E402,F401
