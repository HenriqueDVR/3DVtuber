# Testing OC2VRM

Thank you for trying it. This page gets the app running on your computer and says what kind of
feedback helps. Windows is what it was made and tested on. **Linux** has been run once, in a bare
Debian container without a screen: the launcher, the app, fetching Blender and the add-on, a whole
build of a test character and the test suite all worked; a real desktop has not been tried.
**macOS has never been run**: its steps are written from how the app works, so the first thing worth
reporting there is whether they got you to a running app.

## 1. Get it running

On Windows you don't have to install anything beforehand: the app gets what it needs, asking
before each download.

1. Unzip the copy you were sent and open its folder.
2. Double-click **`OC2VRM.bat`**.
   - No Python on the computer: it offers to install Python 3.12 (through winget, Windows' own
     installer). Answer **Y**.
   - Then it offers Pillow, a small Python package the app needs. Answer **Y**.
3. The app opens on **Setup** while something a build needs is missing. Go down the list: a red line
   has a button that fixes it.
   - **Blender > Get it for me**: downloads Blender 5.2 LTS from blender.org (about 390 MB) into
     the app's own folder and adds the VRM Add-on. A few minutes. Nothing is installed on the
     computer, and a Blender you already have is neither used nor changed unless you pick it.
   - When nothing is red, press **All set: make an avatar**. From then on the app opens on your
     characters.

Nothing is downloaded unless you answer Y or press a button that says so. The same page is
**Setup** in the app's menu at any time. After setup the app is fully local: your
models and avatars stay on your computer.

### macOS

1. Python 3.10 or newer: the installer from [python.org/downloads](https://www.python.org/downloads/)
   brings everything; with Homebrew, `brew install python@3.12 python-tk@3.12`.
2. Blender: download it from [blender.org/download](https://www.blender.org/download/) and drag it
   into **Applications** (the app looks there). The app does not fetch Blender on macOS.
3. Download or clone this repository. Double-click **`OC2VRM.command`** (the first time: right-click,
   Open, Open), or in Terminal: `sh oc2vrm.sh`. It offers to install Pillow.
4. The app opens in your browser on **Setup**. **VRM Add-on > Install it for me** adds it to
   Blender; or in Blender: Edit > Preferences > Get Extensions > "VRM format".

### Linux

1. Python 3.10 or newer with pip and tkinter: `sudo apt install python3 python3-pip python3-tk git`
   (Fedora: `python3-tkinter`; Arch: `python python-pip tk`). On a very bare system Blender also
   wants the usual desktop libraries (Debian: `libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1
   libgl1 libegl1 libxkbcommon0 libsm6 libice6`); a normal desktop has them.
2. Download or clone this repository, then `sh oc2vrm.sh`. It offers to install Pillow (where the
   system refuses pip, it names the system package instead).
3. The app opens in your browser on **Setup**. On a 64-bit Intel/AMD computer
   **Blender > Get it for me** fetches blender.org's build into the app's folder with the VRM
   Add-on. Or install Blender yourself: unpack blender.org's download into `/opt`, your home
   folder, `~/Applications` or `~/.local/share`, or have `blender` on your PATH. Snap and Flatpak
   Blenders are not looked for: add theirs on the Setup page if you use one.

Warudo is a Windows program: on macOS and Linux load the finished `.vrm` in VSeeFace-style apps
that read VRM 1.0, or VNyan under Proton, and skip *Send to Warudo*.

### By hand, any system

| What | Why | How |
|---|---|---|
| **Python 3.10 or newer** (3.12 recommended) | runs the app | [python.org/downloads](https://www.python.org/downloads/); in the installer tick **Add python.exe to PATH**; then `pip install -r requirements.txt` and `python -m app.server` |
| **Blender 4.2 or later** (5.2 LTS is the tested one) | does the 3D work in the background; you never open it | [blender.org/download](https://www.blender.org/download/), then **Check again** |
| **VRM Add-on for Blender** | writes the avatar file | the app's *Install it for me* button, or in Blender: Edit > Preferences > Get Extensions > "VRM format" |

## Getting a fix

When a fix is published the **Setup** page says "A newer one is published" under **This app**. Press
**Update**: it downloads the app's own code (about 8 MB) and starts the app again. Your characters,
their builds and Blender are not touched and nothing else is downloaded again. Say which version you
are on in a report: the line under **This app**.

Keep the file `update.json` that came in your copy: it is what lets the app ask for updates. Don't
share it.

## 2. What to bring

A 3D model of a humanoid character. The tested route:

- Generate it in **Tripo** (P2 works best), standing in a **T-pose** (arms straight out).
- Press **Rig**, pick "VRM 1.0 humanoid", export as **FBX**. Don't use Tripo's rigged GLB.

A plain `.glb` or `.obj` works too (the app places the skeleton itself), and a finished `.vrm`
(VRM 1.0) is taken in as it is and improved.

## 3. What to try

1. **New character**: pick your model, give it a name, press *Create the character*. A minute or two.
2. **Face**: mark the eyes and mouth on the picture (or *Place them for me*, an optional install).
3. **Save and build**: a few minutes. The Build page says what is in the file and what is worth a look.
4. Load the `.vrm` in **Warudo** or **VNyan** (*Send to Warudo* copies it there) and move: raise
   your arms, turn your head, lean, blink, talk, look around.

## 4. What to report

Most useful, in this order:

- **Anything that stopped you**: a step that failed, a button that did nothing, a page you didn't
  understand. Say what you expected.
- **What looked wrong in the tracking app**, and in which pose: a screenshot per pose is worth more
  than a description. Tears, pieces left behind when an arm moves, hair or cloth through the body,
  fingers, eyes, mouth.
- **How long it took** and whether the instructions on each page were enough.

To send a report:

1. Press **Report**: it is on every error the app shows (the red note in the corner, a failed step
   on the Build page), and on the **Stuck?** page for anything else, a build that went well included.
2. Write what happened and what you expected, tick what may go with it, and press **Send**. It
   posts one zip to our private Discord channel with a few lines saying where it stopped. Nothing
   is ever sent without that press. **Only make the zip** writes it
   (`out/<name>/reports/<name>_<date>.zip`) and sends nothing: give it to us yourself.

What is in it: that character's settings, each step's log and timing, the checks' numbers, the app
and Blender versions and what kind of computer it ran on (system, processor, memory, graphics
card). Folder names and your Windows user name are taken out of every file, and so is the author
name set for the avatar. **No picture of your character and no model is in it** unless you tick the
boxes: pictures show us what went wrong; the model you gave the app (with its rig, and your paint)
lets us build your character ourselves with a fix and see whether it worked, which is the most
help of all. A model that is sent has its textures written smaller so the zip fits in a Discord
message; its shape is not changed. A report of a build that went well helps too:
the rules are tuned on few characters, and yours is another.

## Known limits

- Fingers are the weakest part; finger tracking may look stiff or off.
- A model with very loose or layered clothing may need a piece given a role on the Parts page
  (cape, hanging sleeve, hat); the Movement page lists pieces that stay behind.
- The first build of a character is the slow one; later builds restart from the step that changed.
- VRM 0.x files (older VRoid exports) are not taken in: export as VRM 1.0.
