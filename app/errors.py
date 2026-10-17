"""Every way a step fails, every reason the app refuses and every finding of the look before a build
(app/preflight.py), each with a code, what it means, what to do and, where there is one, the option
or the page's action that does it.

A failed step's log is matched against these (a script may also name its code itself with a
`VTUBER_MAKER_ERROR {"code": ..., "detail": ...}` line); a refusal of the engine's own names its code
where it is raised. So the app and agents get the fix from docs/agent/troubleshooting.md instead of a
traceback's last lines, and the page only draws what it is handed: it once read the log's words itself
and told a tester who had marked the face to mark it ("face over the eye patch", read as "no patch").
Codes are stable: the troubleshooting doc and the GUI refer to them.
"""

import json
import re
from collections import namedtuple

# code; the steps whose log it is read in (None: any; (): never read in a log, named where it is
# raised); the pattern in the log; what happened; what to do; the manifest options that mend it, or
# None; the page of the app where it is mended; what that page's button does (jobs and calls the page
# knows, app/gui/js/build.js ACTIONS); and whether the same inputs fail the same way again, so that the
# look before a build can say so (`again`): never for what the computer ran short of.
Known = namedtuple("Known", "code steps pattern what fix options page action again", defaults=(None, None, None, False))

KNOWN = [
    Known("VM-OOM", None, r"Malloc returns null|STOPPED by the engine after \d+ s: out of memory",
          "Blender ran out of memory",
          "close other programs and rerun; a model over ~100k triangles is worth decimating first"),
    Known("VM-DISK-FULL", None, r"STOPPED by the engine after \d+ s: disk full|No space left on device|not enough space on the disk",
          "the drive the build writes to is full",
          "free up space (Characters page: it removes old runs' model files and keeps their reports), then rerun",
          page="home"),
    Known("VM-TIMEOUT", None, r"STOPPED by the engine after \d+ s$",
          "the step ran past the engine's time limit and was stopped",
          "rerun the step alone; if it hangs again, send a bundle"),
    # the operator's own "could not be found", not its name: vrm_check's command line, echoed at
    # the top of its log, names import_scene.vrm, and every failed check read as a missing add-on
    Known("VM-NO-VRM-ADDON", None, r"_scene\.vrm\" error, could not be found|VrmAddonMissing|has no attribute 'vrm'",
          "the VRM Add-on for Blender isn't installed in this Blender",
          "install \"VRM format\" in it (Preferences > Get Extensions); the app turns it on by itself",
          page="setup"),
    Known("VM-EXPORT-POSE", None, r"ExportMismatch",
          "the exported .vrm doesn't hold the rest pose that was built: the VRM Add-on re-posed it on export",
          "use a VRM Add-on version the app was tested with and rerun the step; if it repeats, send a bundle", again=True),
    Known("VM-TRIPO-MIX", None, r"'ShaderNodeMix' object has no attribute 'image'",
          "the model's material multiplies its colour by a grey factor (newer Tripo exports)",
          "run the matte step on the model first (scripts/matte_materials.py)", page="model", again=True),
    Known("VM-LICENCE", ("build",), r"its licence has to be read first",
          "the avatar's licence has to be read before the app changes it",
          "read the terms at the address in brackets; if they let you change the avatar (or it is yours), "
          "say so on the Options page (\"The avatar's licence\") and build again", page="options", again=True),
    Known("VM-ARMS-AT-SIDES", ("survey", "build"), r"the arms don't stand clear of the body",
          "the arms can't be told from the body",
          "generate the model in a T-pose (arms straight out to the sides) and make the character from that file",
          page="model", again=True),
    Known("VM-NOT-HUMANOID", ("survey", "build"), r"doesn't read as a standing humanoid|LandmarkError",
          "joint detection found no standing body",
          "check the model stands upright, alone and unposed in the viewer; a model in a T-pose reads best",
          page="model", again=True),
    Known("VM-ORIENT-UNSURE", ("survey",), r"UNSURE: check the model stands up",
          "the model's orientation couldn't be decided",
          "turn it upright, facing you, in Blender and export it again", page="model", again=True),
    Known("VM-NO-HAIR-BONES", ("hair",), r"need at least one array to stack",
          "no hair bones to reweight to (a rigid head)",
          "rerun: current builds skip hair weights without hair bones"),
    Known("VM-EYES-REFUSED", ("eyes",), r"^FAILED:",
          "the eye step's own check refused the moving eyes (iris over the lids, or hidden by the face)",
          "for a face with little eye white, set eyes to slide; read the eye close-ups in the step's log folder",
          {"eyes": "slide"}, page="face", again=True),
    Known("VM-UNSUPPORTED-FILE", ("survey", "build"), r"Unsupported model type",
          "the model file type can't be read",
          "export it as FBX, GLB/glTF or OBJ; a .zip must be unpacked first, and a path pasted with quotes "
          "around it reads as this too: delete the character (Model page) and make it again from the file itself",
          page="model", again=True),
    Known("VM-BLENDER-OLD", None, r"needs Blender 4\.2 or later",
          "this Blender is older than 4.2, which the pipeline needs",
          "install Blender 4.2 LTS or later and point the app at it ($BLENDER or --blender)", page="setup"),
    # raised by the engine's own check of a step's output, never matched in a log
    Known("VM-BAD-OUTPUT", (), None,
          "the step exited cleanly but its output file is missing, truncated or not a valid VRM 1.0",
          "rerun the step; if it repeats, the step's script has a bug: send a bundle"),
    Known("VM-STEP-DAMAGE", (), None,
          "the step changed something that isn't its own: another step's shape keys, the mesh, or the finger bones",
          "the step's script has a bug: send a bundle (the reason names what changed)", again=True),

    # ----- what the engine refuses before a step starts, named where it is raised
    Known("VM-BUSY", (), None,
          "another build is running: one at a time",
          "wait for it to end, or stop it (Stop, beside its progress); a build that ended badly lets go by itself"),
    Known("VM-NO-SURVEY", (), None,
          "the model hasn't been surveyed, and every step reads the survey",
          "build from the start (Build everything), or run the survey alone first", page="build"),
    Known("VM-NOTHING-BUILT", (), None,
          "the step works on the file the steps before it leave, and none has left one yet",
          "build from the start (Build everything)", page="build"),
    Known("VM-NO-EXPORT", (), None,
          "there is no exported .vrm to check or to send yet",
          "build to the end first (Build everything, or run from \"export\")", page="build"),
    Known("VM-STATE-UNREADABLE", (), None,
          "the record of this character's builds (state.json) can't be read, and neither can its backup",
          "move state.json out of the character's folder: the builds stay where they are, and the next build starts the record afresh"),
    Known("VM-FOLDER-FILE", (), None,
          "the folder avatars are sent to already holds another file under this avatar's name",
          "rename or remove that file there, or pick another folder, then send again", page="build"),
    Known("VM-NO-FOLDER", (), None,
          "no folder to send avatars to has been picked",
          "pick one (your tracking app's avatar folder, for instance): the app remembers it", page="build"),

    # ----- the look before a build (app/preflight.py): what is known to go wrong before a Blender starts
    Known("VM-NO-BLENDER", (), None,
          "Blender wasn't found, or the one chosen doesn't start",
          "get Blender on the Setup page: the app can fetch it for itself", page="setup"),
    Known("VM-NO-MODEL", (), None,
          "the model file this character names isn't there",
          "put the file back where it was, or name the file again on the Model page", page="model"),
    Known("VM-FILE-GONE", (), None,
          "a file this character names isn't there",
          "put it back where it was, or take it off the character"),
    Known("VM-RIG-GLB", (), None,
          "the rig file is a GLB whose bind matrices no skeleton can have (Tripo's rigged GLB): every bone would import broken",
          "export the rigged model from Tripo as FBX and name that file as the model", page="model"),
    Known("VM-NO-TEXTURE", (), None,
          "the model has no colour texture, so hair, face and skin can't be told apart and no face can be made",
          "export the model with its texture (GLB, or FBX with the pictures beside it) and make the character again",
          page="model"),
    Known("VM-ARMS-DOWN", (), None,
          "the arms hang at the sides, where sleeves are fused with the body",
          "generate the model in a T-pose (arms straight out to the sides) and make the character from that file",
          page="model"),
    Known("VM-HEAVY-MODEL", (), None,
          "the model has far more triangles than an avatar needs: every step crawls and tracking apps struggle",
          "export the low-poly version, or decimate it to about 50,000 triangles, and make the character from that file",
          page="model"),
    Known("VM-FRAGMENTS", (), None,
          "the surface is in a great many separate pieces: it tears apart between them, and hair, clothing and fingers can't be read from it",
          "use the generator's own low-poly export; decimating in Blender, merge vertices by distance first (a generated "
          "model is cut along every texture seam, and decimated as it is each patch shrinks away from its neighbours)", page="model"),
    Known("VM-COARSE-MODEL", (), None,
          "the model has too few triangles for a face that bends: only the mouth is moved",
          "use a version of about 50,000 triangles if there is one", page="model"),
    Known("VM-PARTS-NOT-READ", (), None,
          "the model is one fused surface with no pieces to read: the face is taken from the painted texture and there are no eye pieces to blink",
          "nothing to do if it looks right once built; a model generated in parts reads better", page="model"),
    Known("VM-NO-FACE-POINTS", (), None,
          "the eyes and mouth aren't marked: the avatar can't blink, has no face tracking and its eyes don't move",
          "mark them on the Face page", page="face"),
    Known("VM-FACE-FILE", (), None,
          "the face points can't be read",
          "mark the eyes and mouth again on the Face page", page="face"),
    Known("VM-FACE-MARKS", (), None,
          "the face points can't be where they are marked",
          "look at the marks on the Face page and drag the ones that are off", page="face"),
    Known("VM-FACE-REFRAMED", (), None,
          "the face points were marked on another picture of the face than the one the model wears now",
          "mark the eyes and mouth again on the Face page", page="face"),
    Known("VM-ROLE-ON-FACE", (), None,
          "a swinging role is given to the face itself, or to an eye or mouth piece",
          "take the role off that piece on the Parts page", page="parts", action="remove_pieces"),
    Known("VM-HAIR-AS-HAT", (), None,
          "the hair is named a hat, so it is stuck to the head and none of it swings",
          "take the hat role off the hair pieces on the Parts page (or call them hair)", page="parts", action="remove_pieces"),
    Known("VM-LISTING-STALE", (), None,
          "the parts were listed before the eyes were marked, so the roles given can't be checked against the face",
          "list the parts again: a few seconds", page="parts", action="list_pieces"),
    Known("VM-PIECE-GONE", (), None,
          "a piece named on the Parts page isn't on this model (another model, or the same one exported again)",
          "take the role off on the Parts page and give it again to the piece as it is now", page="parts", action="remove_pieces"),
    Known("VM-EYES-MODE", (), None,
          "the way chosen for the eyes to move is one this model can't take",
          "leave the eyes on automatic (Face page)", {"eyes": None}, page="face"),
    Known("VM-EYES-UNMARKED", (), None,
          "eye tracking is chosen but the eyes aren't marked, so it isn't built",
          "mark the eyes on the Face page", page="face"),
    Known("VM-EYES-FALLBACK", (), None,
          "the eyes are built the way an earlier build fell back to, and the face points have changed since",
          "let the build try the moving iris again", page="face", action="forget_eyes_fallback"),
    Known("VM-DISK-LOW", (), None,
          "the drive the build writes to is nearly full",
          "free up space (Characters page: it removes old runs' model files and keeps their reports)", page="home"),
    Known("VM-MEMORY-LOW", (), None,
          "there is less memory to be had than this model's build has taken",
          "close other programs (a browser, a game, another Blender) before building", page="setup"),
    Known("VM-SYSTEM-DISK-LOW", (), None,
          "the system drive is nearly full: Windows can't grow its page file, so Blender may run out of memory",
          "free some space on it", page="setup"),
    Known("VM-BLEND-NEWER", (), None,
          "this build's files were saved by a newer Blender than the one chosen now, which can't open them",
          "build from the start, or choose the newer Blender on the Setup page", page="setup"),
    Known("VM-PATH-LONG", (), None,
          "the character's folder lies so deep on the disk that the files a build writes there pass the 260 characters Windows gives a path",
          "move the app's folder nearer the top of the drive (C:\\OC2VRM), or give the character a shorter name"),
    Known("VM-ONEDRIVE", (), None,
          "the builds are written inside a folder OneDrive keeps in step: it uploads every file of a build, hundreds of megabytes, and holds each while it does",
          "move the app's folder out of OneDrive (to C:\\OC2VRM, say); the characters in its out folder move with it"),
    Known("VM-MODEL-ONLINE", (), None,
          "the model file is kept online only (OneDrive's cloud mark): it has to be downloaded before it can be read",
          "right-click the file and choose \"Always keep on this device\", or copy it out of OneDrive", page="model"),
    Known("VM-NO-BROWSER", (), None,
          "no Edge or Chrome was found: the finished avatar isn't drawn to see what shows through it, and no contact sheets are made",
          "nothing is wrong with the avatar for it; to have those checks, install Edge or Chrome (or name one in VTUBER_MAKER_BROWSER)"),
    # a build whose file went undrawn passed as one with nothing seen: the fox girl kept her moving iris over a
    # socket that shows the inside of her head, as the look that would have sent her back to her discs saw nothing
    Known("VM-NOT-DRAWN", (), None,
          "the finished avatar couldn't be drawn as apps draw it: what shows through its eyes and the eye layers "
          "past the face weren't looked at, and the eyes weren't put back to the discs for it",
          "free space on the drive (the browser needs about 5 GB free where it keeps its profile), then build again "
          "from the exported file's check", page="home"),
]

TAG = "VTUBER_MAKER_ERROR"


# Why the eye step refused, by its own words (its `FAILED:` lines): one code, and what happened said
# for each. The page once told these apart itself, by a word in the log, and took "face over the eye
# patch" for "no face patch": a tester who had marked the face was told to mark it.
# Each: its words in the log, what happened, what to do, the option that does it. The advice never
# names the way that was just refused (a tester's `eyes: slide` failed and was told to set eyes to
# slide); the engine sharpens the patch's when it knows the model has eye pieces (`_eyes_advice`).
EYES_REFUSED = (
    (r"iris shows over the lids", "the blink doesn't cover the moving iris, so eye tracking was left out",
     "try sliding eye patches (Face page), or a fuller blink there", {"eyes": "slide"}),
    (r"face over the eye patch", "the sliding eye patch would go under the face, so eye tracking was left out",
     "try another way for the eyes to move on the Face page, and check the eye marks against the eye close-ups in the step's log folder", None),
    (r"aren't modelled as iris discs", "the eyes aren't modelled as iris discs of their own, so there is nothing of the model's to slide",
     "leave the eyes on automatic (Face page)", {"eyes": None}),
    (r"too little of the iris discs shows", "too little of the model's iris discs shows between the lids to paint whole irises from",
     "leave the eyes on automatic (Face page)", {"eyes": None}),
)
# What a failure with no code is called, and what to do about one: it is the developers' to name.
UNKNOWN = ("it failed in a way the app has no name for",
           "send a report (the Report button): it holds the step's log, which says what went wrong")
# Lines a Blender prints as it ends or as it works, whatever happened in it: never the reason.
CHATTER = ("###", "Blender", "Error: script failed", "Saved session recovery", "LOADED UPDATER", "Info:", "FBX version")
TRACEBACK = "Traceback (most recent call last)"


def classify(step, output):
    """{"code", "what", "fix", "options"} for a failed step's log, or None when it isn't a known one."""
    for line in output.splitlines():
        if line.startswith(TAG):
            try:
                named = json.loads(line[len(TAG):].strip())
            except ValueError:
                break
            known = describe(named.get("code"))
            return {**(known or {"code": named.get("code"), "what": named.get("detail", ""), "fix": "", "options": None}),
                    "detail": named.get("detail", "")}
    for known in KNOWN:
        if known.pattern and (known.steps is None or step in known.steps) and re.search(known.pattern, output, re.MULTILINE):
            return _closer(_said(known), output)
    return None


def _closer(known, output):
    """`known` with what happened said as closely as the log allows: which of the eye step's refusals
    it was, with that one's advice where the code's general advice still stands (the engine may have
    put a sharper one there already)."""
    if known["code"] != "VM-EYES-REFUSED":
        return known
    refused = next((line.strip()[len("FAILED:"):].strip() for line in output.splitlines() if line.strip().startswith("FAILED:")), "")
    found = next((row for row in EYES_REFUSED if re.search(row[0], refused or output)), None)
    if found is None:
        return {**known, **({"detail": refused} if refused else {})}
    general = describe("VM-EYES-REFUSED")
    advice = {"fix": found[2], "options": found[3]} if known.get("fix") == general["fix"] else {}
    return {**known, "what": found[1], **advice, **({"detail": refused} if refused else {})}


def last_line(text):
    """The line of a failed step's output that says why: what a Python traceback ends on (the error
    and its words, without the module it lives in), else the last line that isn't Blender's own talk.

    Blender prints on after a script has fallen over: a tester's build that failed on "the face patch
    has no polygons" ended its log with two lines of the FBX importer's and "Info: Baking map saved to
    internal image", and the last line by itself was that."""
    lines = str(text or "").splitlines()
    raised = max((i for i, line in enumerate(lines) if line.strip().startswith(TRACEBACK)), default=None)
    if raised is not None:
        # after the traceback's own lines (indented) comes the error: "package.module.SomeError: its words"
        ended = next((line.strip() for line in lines[raised + 1:] if line.strip() and not line.startswith((" ", "\t"))), "")
        kind, colon, words = ended.partition(": ")
        if ended and " " not in kind:
            return f"{kind.rsplit('.', 1)[-1]}{colon}{words}"
    return next((line.strip() for line in reversed(lines) if line.strip() and not line.strip().startswith(CHATTER)), "")


def failure(step, known, said):
    """A failure as the page draws it, with nothing left for the page to work out: {"code", "what",
    "detail", "fix", "options", "page", "action"}. `known`: its code as the engine kept it, or None;
    `said`: the failure's own words (the error's message, which ends with the log's end).

    A failure the app has no name for has no code: what it says is the log's last telling line, and
    the way on is a report. A state.json from before failures were kept this way holds `known` (or
    nothing) and the words: read through here it is drawn the same."""
    if known and known.get("code"):
        known = _closer(dict(known), said) if not known.get("detail") else known
        return {"code": known["code"], "what": known.get("what", ""), "detail": known.get("detail", ""),
                "fix": known.get("fix", ""), "options": known.get("options"),
                "page": known.get("page") or (describe(known["code"]) or {}).get("page"),
                "action": known.get("action")}
    found = classify(step, said) if step else None
    if found:
        return failure(step, found, said)
    return {"code": None, "what": UNKNOWN[0], "detail": last_line(said), "fix": UNKNOWN[1], "options": None, "page": None, "action": None}


def describe(code):
    return next((_said(known) for known in KNOWN if known.code == code), None)


def repeats(code):
    """Whether a failure with this code comes again from the same inputs (see `Known.again`)."""
    return any(known.again for known in KNOWN if known.code == code)


def _said(known):
    """A code as the engine keeps it with a failure and the page draws it. `page` and `action` only
    where there is one: a state.json written before they existed has neither."""
    return {"code": known.code, "what": known.what, "fix": known.fix, "options": known.options,
            **({"page": known.page} if known.page else {}), **({"action": known.action} if known.action else {})}
