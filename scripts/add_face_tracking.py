"""Add ARKit face shapes and a partial blink for webcam face tracking, then export VRM.

    blender -b in.blend -P scripts/add_face_tracking.py -- out.blend out.vrm [--blink 0.4] [--jaw 0.008] [--mouth-only] [--renders dir]

Face trackers (Warudo's MediaPipe, iPhone ARKit) drive ARKit-named blend shapes; the build only
makes the vowels aa/ih/ou/ee/oh for lip sync, which open the painted mouth but leave the face
still. These shapes move the face around the mouth: the jaw and chin drop, lip corners rise,
fall, pull in or slide sideways, blended with the vowels for the mouth opening itself. With
`--mouth-only` only the mouth patch moves: it opens with the vowels, slides sideways and curls
its corners over the face, which stays still (for a coarse face, like a mascot's, that dents
when it bends). Build with `make_vrm.py --free-mouth` so the closed mouth is part of the patch,
not painted on, and moves too.

The blink lowers only the upper half of each eye onto a line below its middle, fading out
before the brow, so the brow and lower lid stay put; painted eyes close from both sides. `--blink` is how far it closes (1 =
upper half flat). At 0 every shape is the model as it was. The VRM blink expressions are bound
to the new keys. The original file is left alone. Needs the VRM Add-on for Blender.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from vtuber_maker.pipeline import eyes, face_rig, fingers, rig, vrm_export  # noqa: E402

# Lip corner moves, as (x outward, y forward, z up) in mouth widths, with the vowel mixed in.
CORNERS = {
    "mouthSmile": ((0.12, 0.0, 0.2), ("ee", 0.4)),
    "mouthFrown": ((0.04, 0.0, -0.2), None),
    "mouthStretch": ((0.15, 0.0, -0.04), ("ih", 0.5)),
}
CORNER_REACH = 0.9      # radius of a corner move, in mouth widths
# --mouth-only: how far the mouth slides sideways and its corners curl up or down, in mouth widths
FREE_SLIDE = 0.5
FREE_SMILE = 0.14
FREE_FROWN = 0.1
SIDE_SHIFT = 0.6        # mouthLeft/Right, in mouth widths
PUCKER_IN = 0.3         # corners pulled towards the middle, in mouth widths
JAW_PIVOT_BACK = 0.1    # jaw hinge behind the mouth (m)
JAW_PIVOT_UP = 0.04     # and above it (m)
CHIN_BEHIND = 0.3       # the chin lies no further behind the mouth than this share of the way to the hinge
JAW_WIDTH = (0.03, 0.075)  # jaw moves fully within this distance of the middle, none past the second (m)
JAW_VOWEL = 1.0         # share of the "aa" vowel mixed into jawOpen
# Blink: the part of the eye above CLOSE_LINE (0 bottom of the insert, 1 top) closes onto it;
# the skin above the lashes follows over face_rig.LID_FADE of the eye height, then nothing moves.
# Eye inserts close fully within face_rig.EYE_WIDTH[0] half widths and not at all past [1].
# Painted eyes (face points given to the build) close from both sides instead, by
# face_rig.squeeze_shut, face_rig.PAINTED_BLINK of the way by default.
CLOSE_LINE = 0.4
# Surfaces within this depth (m) of the front of the painted eye blink with it.
PAINT_DEPTH = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_blend")
    ap.add_argument("out_vrm")
    ap.add_argument("--blink", type=float, help="how far the eye closes, 0 to 1 (default 0.4, or 0.7 for painted eyes, which close from both sides)")
    ap.add_argument("--jaw", type=float, default=0.008, help="how far the chin drops on jawOpen (m)")
    ap.add_argument("--mouth-only", action="store_true",
                    help="mouth shapes open only the mouth, from the vowels; the face around it stays still")
    ap.add_argument("--no-blink", action="store_true", help="the eyes are hidden (a visor, glasses): mouth shapes only")
    ap.add_argument("--renders", type=Path, help="write face close-ups of each shape here")
    ap.add_argument("--blink-tries", type=float, nargs="*", default=[], help="also render these blink amounts")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm)
    shapes = FaceShapes(obj)
    if args.mouth_only:
        shapes.add_mouth_from_vowels()
    else:
        shapes.add_mouth(args.jaw)
    if not args.no_blink:
        shapes.add_blink("", args.blink if args.blink is not None else (face_rig.PAINTED_BLINK if shapes.painted else 0.4))
        bind_blink(arm, obj)

    bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(args.out_blend), copy=True)
    if not vrm_export.skipped():
        print("exported", bpy.ops.export_scene.vrm(filepath=bpy.path.abspath(args.out_vrm), armature_object_name=arm.name))

    if args.renders:
        for amount in args.blink_tries:
            shapes.add_blink(f"_try{amount}", amount)
        render(obj, shapes, args.renders, args.blink_tries)


class FaceShapes:
    def __init__(self, obj):
        self.obj = obj
        mesh = obj.data
        self.keys = mesh.shape_keys.key_blocks
        self.count = len(mesh.vertices)
        self.basis = self.read("Basis")
        inserts = self.point_attribute(face_rig.EYE_INSERT_ATTRIBUTE)
        # iris layers, eye whites and sliding eye patches (eyes.py) move on their own, never with the face
        self.iris = (self.point_attribute(eyes.IRIS_ATTRIBUTE, missing=0) > 0) | (self.point_attribute(eyes.WHITE_ATTRIBUTE, missing=0) > 0)
        self.patches = self.point_attribute(eyes.PATCH_ATTRIBUTE, missing=0)
        self.fixed = self.point_attribute(face_rig.FIXED_ATTRIBUTE) != 0
        self.movable = (~self.fixed & ~self.iris & (self.patches == 0)).astype(float)
        # vertices swung by spring chains (hair, cloth, appendages, a hat's tip): not face
        humanoid = set(rig.BONE_NAMES.values()) | {fingers.bone_name(side, f, seg) for side in ("left", "right")
                                                    for f in fingers.FINGERS for seg in fingers._segments(f)}
        chain = {g.index for g in obj.vertex_groups if g.name not in humanoid and obj.parent is not None
                 and g.name in obj.parent.data.bones}
        self.on_springs = np.array([sum(g.weight for g in v.groups if g.group in chain) > 0.1 for v in mesh.vertices])
        self.eyes = {side: self.basis[inserts == ident] for side, ident in face_rig.EYE_INSERT_IDS.items()}
        painted = obj.get(face_rig.PAINTED_EYES_PROPERTY)
        # modelled eyes blink with their lids even when face points were given: the painted squeeze
        # closes 70% and a moving iris (add_eye_tracking) showed through the gap on model A, where the lid
        # blink her tracked build had passes
        if (inserts > 0).any():
            painted = None
        self.painted = painted is not None
        if painted is not None:
            # the painted eyes, where face points put them: the inserts can sit elsewhere
            self.eyes = {side: np.array([[x0, 0.0, z0], [x1, 0.0, z1]]) for side, (x0, z0, x1, z1) in painted.items()}
        cover = obj["vtuber_maker_mouth_cover"]
        self.mouth = np.array([cover["x"], cover["y"], cover["z"]])
        self.width = float(cover["width"])

    def read(self, name):
        co = np.empty(self.count * 3)
        self.keys[name].data.foreach_get("co", co)
        return co.reshape(-1, 3)

    def point_attribute(self, name, missing=None):
        if missing is not None and name not in self.obj.data.attributes:
            return np.full(self.count, missing, dtype=np.int32)
        values = np.empty(self.count, dtype=np.int32)
        self.obj.data.attributes[name].data.foreach_get("value", values)
        return values

    def add(self, name, delta, movable=None):
        if name in self.keys:
            self.obj.shape_key_remove(self.keys[name])
            self.keys = self.obj.data.shape_keys.key_blocks
        movable = self.movable if movable is None else movable
        face_rig._add_key(self.obj, name, self.basis + delta * movable[:, None])

    def vowel(self, name, share):
        return (self.read(name) - self.basis) * share

    def bump(self, centre, reach, offset):
        """Move vertices near centre by offset, fading smoothly to nothing at reach."""
        distance = np.linalg.norm(self.basis - centre, axis=1)
        weight = face_rig._fade(distance, 0.0, reach)
        return weight[:, None] * np.asarray(offset)[None, :]

    def side_mask(self, side):
        """1 on the character's side of the mouth (its left is +X), blending across the middle."""
        sign = 1 if side == "Left" else -1
        t = np.clip(sign * (self.basis[:, 0] - self.mouth[0]) / (0.3 * self.width) + 0.5, 0, 1)
        return (t * t * (3 - 2 * t))[:, None]

    def add_mouth(self, jaw_drop):
        w = self.width
        corners = {"Left": self.mouth + [w / 2, 0, 0], "Right": self.mouth - [w / 2, 0, 0]}
        reach = CORNER_REACH * w

        # jaw: everything below the mouth turns down about a hinge behind and above it
        pivot = self.mouth + [0, JAW_PIVOT_BACK, JAW_PIVOT_UP]
        chin = np.array([self.mouth[0], self.mouth[1], self.basis[self.movable > 0, 2].min()])
        angle = jaw_drop / np.linalg.norm((chin - pivot)[1:])
        r = self.basis - pivot
        turned = r.copy()
        # rotating about X; the sign that sends the chin (below and in front of the hinge) down
        turned[:, 1] = r[:, 1] * math.cos(angle) - r[:, 2] * math.sin(angle)
        turned[:, 2] = r[:, 1] * math.sin(angle) + r[:, 2] * math.cos(angle)
        # 1 from a little below the mouth line down, 0 above it
        below = 1 - face_rig._fade(self.mouth[2] - self.basis[:, 2], -0.1 * w, 0.4 * w)
        sideways = face_rig._fade(np.abs(self.basis[:, 0] - self.mouth[0]), *JAW_WIDTH)
        jaw = (turned - r) * (below * sideways)[:, None]
        # the chin: the lowest point before the hinge, on the middle. The neck can reach lower behind
        # the hinge, and rises a little as the jaw turns (model A's T-pose FBX: 2 cm behind it); so does
        # the front of a long neck nearly under it (model D), which is not the chin either
        chin_side = ((self.movable > 0) & (self.basis[:, 1] < self.mouth[1] + CHIN_BEHIND * JAW_PIVOT_BACK)
                     & (np.abs(self.basis[:, 0] - self.mouth[0]) < JAW_WIDTH[0]))
        lowest = np.argmin(np.where(chin_side, self.basis[:, 2], np.inf))
        if jaw[lowest, 2] >= 0:
            raise SystemExit("jaw turns the wrong way")
        self.add("jawOpen", jaw + self.vowel("aa", JAW_VOWEL))

        for name, (move, vowel) in CORNERS.items():
            for side, corner in corners.items():
                outward = 1 if side == "Left" else -1
                offset = np.array([outward * move[0], -move[1], move[2]]) * w
                delta = self.bump(corner, reach, offset)
                if vowel:
                    delta += self.vowel(*vowel) * self.side_mask(side)
                self.add(f"{name}{side}", delta)

        pucker = sum(self.bump(c, reach, [(-1 if s == "Left" else 1) * PUCKER_IN * w, -0.1 * w, 0]) for s, c in corners.items())
        self.add("mouthPucker", pucker + self.vowel("ou", 0.7))
        self.add("mouthFunnel", 0.6 * pucker + self.vowel("oh", 1.0))
        for side in ("Left", "Right"):
            sign = 1 if side == "Left" else -1
            self.add(f"mouth{side}", self.bump(self.mouth, 1.6 * w, [sign * SIDE_SHIFT * w, 0, 0]))
        print("added mouth shapes")

    def add_mouth_from_vowels(self):
        """ARKit mouth shapes that move only the mouth patch, for faces too coarse to bend round it.

        The opening comes from the vowels; sideways slides and curled corners move the patch
        over the face, following its curve. The face itself never moves.
        """
        grid = np.empty(self.count * 3)
        self.obj.data.attributes[face_rig.MOUTH_GRID_ATTRIBUTE].data.foreach_get("vector", grid)
        patch = grid.reshape(-1, 3)[:, 2] > 0.5
        mesh = self.obj.data
        polygons = [p.vertices[:] for p in mesh.polygons if not patch[list(p.vertices)].all()]
        face = BVHTree.FromPolygons([Vector(v) for v in self.basis], polygons)
        w = self.width

        def on_patch(move):
            """Delta moving the patch's rest positions by move(points), kept on the face."""
            moved = self.basis.copy()
            points = move(self.basis[patch].copy())
            for point in points:
                hit = face.ray_cast(Vector((point[0], point[1] - 0.03, point[2])), Vector((0.0, 1.0, 0.0)), 0.06)[0]
                if hit is not None:
                    point[1] = hit.y - face_rig.MOUTH_SURFACE_GAP
            moved[patch] = points
            return moved - self.basis

        def curl(side, lift):
            sign = 1 if side == "Left" else -1
            reach = face_rig.MOUTH_REST_SHAPE[0] * w

            def move(points):
                t = np.clip(sign * (points[:, 0] - self.mouth[0]) / reach, 0.0, 1.0)
                points[:, 2] += lift * w * t * t
                return points
            return move

        self.add("jawOpen", self.vowel("aa", 1.0))
        self.add("mouthFunnel", self.vowel("oh", 1.0))
        self.add("mouthPucker", self.vowel("ou", 1.0))
        for side in ("Left", "Right"):
            sign = 1 if side == "Left" else -1
            mask = self.side_mask(side)
            self.add(f"mouthSmile{side}", on_patch(curl(side, FREE_SMILE)) + self.vowel("ee", 0.3) * mask)
            self.add(f"mouthFrown{side}", on_patch(curl(side, -FREE_FROWN)))
            self.add(f"mouthStretch{side}", self.vowel("ih", 0.5) * mask)

            def slide(points, sign=sign):
                points[:, 0] += sign * FREE_SLIDE * w
                return points
            self.add(f"mouth{side}", on_patch(slide))
        print("added mouth shapes that move only the mouth")

    def add_blink(self, suffix, amount):
        if self.painted:
            return self.add_blink_meet(suffix, amount)
        for side, points in self.eyes.items():
            low, high = points.min(axis=0), points.max(axis=0)
            centre, half = (low + high) / 2, (high - low) / 2
            height = high[2] - low[2]
            line, top = low[2] + CLOSE_LINE * height, high[2]
            z = self.basis[:, 2]
            across = face_rig._fade(np.abs(self.basis[:, 0] - centre[0]) / half[0], *face_rig.EYE_WIDTH)
            drop = np.where(z > line, amount * (np.minimum(z, top) - line), 0.0)
            above = z > top
            drop[above] *= face_rig._fade(z[above] - top, 0.0, face_rig.LID_FADE * height)
            delta = np.zeros_like(self.basis)
            delta[:, 2] = -drop * across
            name = f"eyeBlink{side.title()}{suffix}"
            self.add(name, delta)
        print(f"added blink{suffix}: closes {amount:.0%}")

    def add_blink_meet(self, suffix, amount):
        """Painted eyes close from both sides: the lid comes down and the bottom comes up.

        The eye squeezes towards face_rig.MEET_LINE of its height by `amount` (1 = shut to a
        line); see face_rig.squeeze_shut. With sliding eye patches (eyes.py) only they squeeze:
        the paint under them is gone and the face stays still.
        """
        for side, points in self.eyes.items():
            low, high = points.min(axis=0), points.max(axis=0)
            centre, half = (low + high) / 2, (high - low) / 2
            z = self.basis[:, 2]
            delta = np.zeros_like(self.basis)
            delta[:, 2] = face_rig.squeeze_shut(self.basis, (low[0], low[2], high[0], high[2]), amount)
            if self.patches.any():
                movable = (self.patches == face_rig.EYE_INSERT_IDS[side]).astype(float)
            else:
                # the paint can be on more than the face part (a face patch over a textured head):
                # move the surface at the front there, but never an iris layer, nor hair: "fixed or
                # not" squeezed model A's fringe strands (about 500 fixed and 100 hair-chain vertices per
                # eye) down with her lids
                keep = ~self.iris & ~self.fixed & ~self.on_springs
                near = (np.abs(self.basis[:, 0] - centre[0]) < half[0]) & (np.abs(z - centre[2]) < half[2]) & keep
                front = self.basis[near, 1].min() if near.any() else self.basis[:, 1].min()
                movable = np.maximum(self.movable, (self.basis[:, 1] < front + PAINT_DEPTH).astype(float)) * keep
            self.add(f"eyeBlink{side.title()}{suffix}", delta, movable)
        print(f"added blink{suffix}: the eyes close {amount:.0%} of the way, lid and bottom meeting")


def bind_blink(arm, obj):
    presets = arm.data.vrm_addon_extension.vrm1.expressions.preset
    for preset, names in {"blink": ("eyeBlinkLeft", "eyeBlinkRight"), "blink_left": ("eyeBlinkLeft",), "blink_right": ("eyeBlinkRight",)}.items():
        expression = getattr(presets, preset)
        expression.morph_target_binds.clear()
        for name in names:
            bind = expression.morph_target_binds.add()
            bind.node.mesh_object_name = obj.name
            bind.index = name
            bind.weight = 1.0
    print("bound blink expressions")


def render(obj, shapes, out_dir, blink_tries):
    """Flat-lit front close-ups of the face at rest and with each new shape."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = 768
    left, right = shapes.eyes["left"].mean(axis=0), shapes.eyes["right"].mean(axis=0)
    spread = float(abs(left[0] - right[0]))
    data = bpy.data.cameras.new("face")
    data.type = "ORTHO"
    data.ortho_scale = 2.2 * spread
    camera = bpy.data.objects.new("face", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = obj.matrix_world @ Vector(((left[0] + right[0]) / 2, -1.0, left[2] - 0.4 * spread))
    camera.rotation_euler = (math.pi / 2, 0, 0)
    keys = obj.data.shape_keys.key_blocks
    views = {"rest": {}, "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}}
    views.update({f"blink_{a}": {f"eyeBlinkLeft_try{a}": 1.0, f"eyeBlinkRight_try{a}": 1.0} for a in blink_tries})
    views.update({k.name: {k.name: 1.0} for k in keys if k.name[0].islower() and "Blink" not in k.name and len(k.name) > 2})
    views["talk_smile"] = {"jawOpen": 0.6, "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    for shown, values in views.items():
        for key in keys:
            key.value = values.get(key.name, 0.0)
        scene.render.filepath = str((out_dir / f"{shown}.png").resolve())
        bpy.ops.render.render(write_still=True)
    for key in keys:
        key.value = 0.0
    print("rendered close-ups to", out_dir)


if __name__ == "__main__":  # scripts/fix_mouth.py rebuilds the mouth shapes with FaceShapes
    main()
