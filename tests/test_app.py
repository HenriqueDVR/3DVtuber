"""The app's engine plans a character's steps from its manifest, and its server hands out only what it should."""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# One Blender a step and one at a time in these tests, as a build was before 2026-10-12: they stand in
# for a Blender at `engine._blender` or at Popen, each step alone. The Blender a run keeps for its steps
# (engine._Worker) and the last steps side by side have tests of their own (class Blenders) that say so.
os.environ["OC2VRM_ONE_BLENDER"], os.environ["OC2VRM_BLENDERS"] = "0", "1"

from app import cli, engine, errors, paint, server, slim, update, vendor  # noqa: E402
from app.engine import Project, manifest  # noqa: E402


def mock_textures(report, sizes):
    from unittest import mock
    return mock.patch.object(report, "TEXTURES", sizes)


class Engine(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "character.json"
        data = manifest.new("model.glb", "Test")
        data["out_dir"] = "out"
        data["face"] = "face.json"
        manifest.save(data, self.path)
        # the machine running the tests may itself be short of disk or memory, and have a Blender chosen
        self.real_machine = engine.free_disk, engine.free_memory, engine.SETTINGS
        engine.free_disk, engine.free_memory, engine.SETTINGS = lambda path: 100 << 30, lambda: 16 << 30, self.root / "settings.json"

    def tearDown(self):
        engine.free_disk, engine.free_memory, engine.SETTINGS = self.real_machine
        self.folder.cleanup()

    def project(self, survey=None, **options):
        data = manifest.load(self.path)
        data["options"].update(options)
        manifest.save(data, self.path)
        project = Project(self.path, blender="blender")
        if survey is not None:
            report = self.root / "survey.json"
            report.write_text(json.dumps(survey))
            project.state["steps"]["survey"] = {"status": "done", "json": str(report)}
        return project

    def runs(self, project):
        return {name for name, runs, _ in project.plan() if runs}

    def test_survey_decides_what_was_left_to_it(self):
        project = self.project({"options": {"rigid_head": True, "cloth_physics": True, "mouth_color": "skin"},
                                "triangles": 40000, "face_patch": True})
        o = project.options()
        self.assertTrue(o["rigid_head"] and o["cloth"] and o["mouth_only"] and o["free_mouth"])
        self.assertEqual(o["mouth_colour"], "skin")
        self.assertNotIn("ahoge", self.runs(project))  # a rigid head has no ahoge

    def test_a_bundle_of_a_build_that_failed_shows_the_model_and_the_failure(self):
        # a tester's bundle of a character that would not build held no picture of it and no build
        # log: the pictures were only the renders of a finished build, the log only a done step's
        project = self.project({"options": {}, "triangles": 40000})
        views = project.out / "face" / "views"
        views.mkdir(parents=True)
        for name in ("color_head_000.png", "color_torso_000.png", "color_torso_180.png", "color_torso_045.png", "z_head_000.exr"):
            (views / name).write_bytes(b"picture")
        log = self.root / "01_build.log"
        log.write_text("the face patch has no polygons")
        project.state["steps"]["build"] = {"status": "not run", "last_failure": {"reason": "...", "log": str(log)}}
        project._save_state()
        found = cli.cmd_bundle(SimpleNamespace(manifest=self.path, blender="blender", note="It won't build"))
        folder = Path(found["bundle"])
        self.assertEqual(sorted(p.name for p in (folder / "model").iterdir()),
                         ["color_head_000.png", "color_torso_000.png", "color_torso_180.png"])
        self.assertEqual((folder / "build_failed.log").read_text(), "the face patch has no polygons")

    def test_a_report_holds_no_picture_and_no_place_on_the_computer_unless_asked(self):
        # for the developers, from a tester's computer: numbers and text, the folders and the user name left behind
        import zipfile
        from app import report
        project = self.project({"options": {}, "triangles": 40000}, hat_tip=True)
        home = self.root / "Users" / "Ayumi"
        run = project.out / "runs" / "20261009-120000"
        (run / "13_check_spots").mkdir(parents=True)
        (project.out / "face" / "views").mkdir(parents=True)
        (project.out / "face" / "views" / "color_torso_000.png").write_bytes(b"picture")
        (run / "13_check_spots" / "stretch.png").write_bytes(b"picture")
        (run / "01_build.blend").write_bytes(b"model")
        (run / "14_export.vrm").write_bytes(b"model")
        slashed = str(run).replace(chr(92), "/")
        (run / "01_build.log").write_text(f"Read {home / 'art' / 'oc.fbx'}, saved {slashed}/01_build.blend, by ayumi")
        (run / "12_mouth_fingerprint.json").write_text(json.dumps({"keys": {}, "blend": str(run / "12_mouth.blend")}))
        project.state["steps"]["build"] = {"status": "done", "seconds": 40, "log": str(run / "01_build.log"), "blend": str(run / "01_build.blend")}
        project.state["steps"]["export"] = {"status": "done", "seconds": 5, "vrm": str(run / "14_export.vrm")}
        project._save_state()
        made = report.make(self.path, note=f"it broke in {home}", blender="blender")
        with zipfile.ZipFile(made["report"]) as archive:
            names = archive.namelist()
            text = {name: archive.read(name).decode() for name in names}
        self.assertEqual(made["left_out"], ["pictures of the character", "the model"])
        self.assertEqual({Path(name).suffix for name in names}, {".json", ".log", ".txt"})
        self.assertIn("runs/20261009-120000/12_mouth_fingerprint.json", names)
        summary = json.loads(text["report.json"])
        self.assertEqual(summary["character"]["options_changed"], {"hat_tip": True})
        self.assertEqual([(s["step"], s["seconds"]) for s in summary["steps"] if s["seconds"]], [("build", 40), ("export", 5)])
        clean = report.scrubber(project, home=home)
        self.assertEqual(clean(f"Read {home / 'art'} by ayumi, {json.dumps(str(home))}"),
                         "Read ~" + chr(92) + 'art by user, "~"')
        self.assertEqual(clean('{"author": "Ayumi Art"}'), '{"author": "(left out)"}')
        for name, body in text.items():
            self.assertNotIn(str(self.root), body, name)
            self.assertNotIn(json.dumps(str(self.root))[1:-1], body, name)
        (self.root / "model.glb").write_bytes(b"the model as it was given")
        asked = report.make(self.path, pictures=True, model=True, blender="blender")
        with zipfile.ZipFile(asked["report"]) as archive:
            names = archive.namelist()
        self.assertIn("model_views/color_torso_000.png", names)
        self.assertIn("runs/20261009-120000/13_check_spots/stretch.png", names)
        # the model is the file the build reads, as it is: with it the character can be built again
        self.assertIn("model/source/model.glb", names)
        self.assertFalse([name for name in names if name.endswith((".vrm", ".blend"))])
        self.assertEqual(asked["left_out"], [])

    def test_a_report_is_sent_only_to_the_address_kept_and_without_the_model_unless_asked(self):
        import zipfile
        from unittest import mock
        from app import report, update
        project = self.project({"options": {}, "triangles": 40000})
        run = project.out / "runs" / "20261009-120000"
        run.mkdir(parents=True)
        (run / "14_export.vrm").write_bytes(b"model")
        project.state["steps"]["export"] = {"status": "done", "vrm": str(run / "14_export.vrm")}
        project._save_state()
        posted = []

        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b""

        def open_url(request, timeout=None):
            posted.append(request)
            return Answer()

        shipped = self.root / "reports.json"
        with mock.patch.object(update, "CONFIG", self.root / "update.json"), mock.patch.object(report, "SHIPPED", shipped):
            # no address: nothing leaves, and the page is told to hand the zip over
            with self.assertRaisesRegex(report.NotSent, "no address"):
                report.send(self.path, note="x", blender="blender", open_url=open_url)
            # only a Discord webhook is an address, whatever the file says
            with self.assertRaisesRegex(ValueError, "Discord webhook"):
                report.use_address("https://example.com/api/webhooks/1/abc")
            (self.root / "update.json").write_text(json.dumps({"repo": "a/b", "reports": "https://example.com/upload"}))
            self.assertIsNone(report.address())
            # the address that comes with the app is the one used; update.json's, when it names one, goes before it
            shipped.write_text(json.dumps({"address": "https://discord.com/api/webhooks/9/shipped"}))
            self.assertEqual(report.address(), "https://discord.com/api/webhooks/9/shipped")
            report.use_address("https://discord.com/api/webhooks/123/abc-DEF")
            self.assertEqual(json.loads((self.root / "update.json").read_text())["repo"], "a/b")
            made = report.send(self.path, note="@everyone it broke", pictures=True, blender="blender", open_url=open_url)
        self.assertEqual(posted, posted[:1])
        self.assertEqual(posted[0].full_url, "https://discord.com/api/webhooks/123/abc-DEF")
        body = posted[0].data
        self.assertIn(b'"allowed_mentions": {"parse": []}', body)
        self.assertIn(b"it broke", body)
        self.assertIn(Path(made["report"]).read_bytes(), body)
        with zipfile.ZipFile(made["report"]) as archive:
            self.assertFalse([name for name in archive.namelist() if name.startswith("model/") or name.endswith(".vrm")])
        self.assertTrue(made["sent"])

    def test_a_model_that_is_sent_has_its_textures_made_smaller_until_it_fits(self):
        # Tripo's 4096 px PNGs are most of its files: model A's 29 MB against the 9 a message takes
        import io
        import random
        import zipfile
        from app import report
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not in this Python: run with python -m unittest tests.test_app")
        noise = random.Random(1)
        picture = Image.frombytes("RGB", (512, 512), bytes(noise.getrandbits(8) for _ in range(512 * 512 * 3)))
        png = io.BytesIO()
        picture.save(png, "PNG")
        shape = bytes(range(256)) * 4
        gltf = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(png.getvalue()) + len(shape)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(png.getvalue())},
                                {"buffer": 0, "byteOffset": len(png.getvalue()), "byteLength": len(shape)}],
                "images": [{"bufferView": 0, "mimeType": "image/png"}]}
        source = Engine.write_glb(self, "model.glb", gltf, png.getvalue() + shape)
        self.project({"options": {}, "triangles": 40000})

        def read(raw):
            length = struct.unpack_from("<I", raw, 12)[0]
            doc = json.loads(raw[20:20 + length])
            binary = raw[28 + length:]
            return doc, [binary[v["byteOffset"]:v["byteOffset"] + v["byteLength"]] for v in doc["bufferViews"]]

        doc, (small, kept) = read(report.smaller_model(source, 128, 85))
        self.assertEqual(kept, shape)
        self.assertEqual(doc["images"][0]["mimeType"], "image/jpeg")
        self.assertEqual(Image.open(io.BytesIO(small)).size, (128, 128))
        self.assertEqual(struct.unpack_from("<I", report.smaller_model(source, 128, 85), 8)[0], len(report.smaller_model(source, 128, 85)))
        # room for the whole file: the first size; room for less: the next that fits; none: said, and no zip left
        whole = report.make(self.path, model=True, blender="blender", fit=4 * 2**20)
        with zipfile.ZipFile(whole["report"]) as archive:
            self.assertEqual(json.loads(archive.read("report.json"))["model"]["textures"], report.TEXTURES[0][0])
            self.assertEqual(read(archive.read("model/source/model.glb"))[1][1], shape)
        with mock_textures(report, ((512, 85), (64, 80))):
            tight = report.make(self.path, model=True, blender="blender", fit=report.SUMMARY_ROOM + 60_000)
            with zipfile.ZipFile(tight["report"]) as archive:
                self.assertEqual(json.loads(archive.read("report.json"))["model"]["textures"], 64)
            before = sorted((self.root / "out" / "reports").iterdir())
            with self.assertRaisesRegex(report.NotSent, "doesn't fit"):
                report.make(self.path, model=True, blender="blender", fit=report.SUMMARY_ROOM + 300)
            self.assertEqual(sorted((self.root / "out" / "reports").iterdir()), before)

    def test_a_report_says_first_where_it_stopped_and_what_the_page_showed(self):
        # a zip alone had to be opened to learn whether it was a failed install or a sleeve
        import zipfile
        from app import report
        project = self.project({"options": {}, "triangles": 40000}, hat_tip=True)
        project.state["steps"]["survey"]["seconds"] = 6
        project.state["steps"]["build"] = {"status": "failed", "reason": f"build: Blender ran out of memory (see {self.root / 'x.log'})\nMalloc",
                                           "known": {"code": "VM-OOM", "what": "Blender ran out of memory", "fix": "close other programs"}}
        project._save_state()
        made = report.make(self.path, note="it stopped", about="The run failed: build\nmore", blender="blender")
        lines = made["headline"]
        self.assertTrue(lines[0].startswith("**Test** | "), lines[0])
        self.assertIn("Failed at **build** [VM-OOM]: build: Blender ran out of memory", lines[1])
        self.assertNotIn(str(self.root), "\n".join(lines))
        self.assertIn("Options changed: hat_tip=True", lines)
        self.assertEqual(lines[-1], "On their page: The run failed: build")
        with zipfile.ZipFile(made["report"]) as archive:
            self.assertEqual(json.loads(archive.read("report.json"))["headline"], lines)
            self.assertIn("On the page: The run failed: build", archive.read("NOTE.txt").decode())

    def test_a_report_can_be_made_with_no_character_open(self):
        # what goes wrong before a character can be opened (an install, the app itself) had no report at all
        import zipfile
        from unittest import mock
        from app import report
        # (a check as engine.setup writes one: its words are "text". Handed over here as "what", which no
        # check has, the test passed while every real report's line read "Setup: None")
        with mock.patch.object(report, "REPO", self.root), mock.patch.object(engine, "setup", lambda: {"version": "5.2.2", "checks": [
                {"name": "VRM Add-on", "state": "bad", "text": "the VRM Add-on isn't installed", "fix": "..."}]}):
            (self.root / "out").mkdir(exist_ok=True)
            (self.root / "out" / "app_errors.log").write_text(f"Traceback in {self.root}")
            made = report.make(None, note="won't start", about="install failed")
        self.assertEqual(made["headline"][0].split(" | ")[0], "**No character open**")
        self.assertIn("Setup: the VRM Add-on isn't installed", made["headline"])
        self.assertTrue(all({"name", "state", "text", "fix"} <= set(check) for check in engine.setup()["checks"]))
        with zipfile.ZipFile(made["report"]) as archive:
            self.assertEqual(sorted(archive.namelist()), ["NOTE.txt", "app_errors.log", "report.json"])
            self.assertNotIn(str(self.root), archive.read("app_errors.log").decode())

    def test_a_report_that_came_in_is_checked_before_it_is_opened(self):
        # the address reports go to is in every tester's copy: what arrives may be from anyone
        import zipfile
        from unittest import mock
        from app import report
        project = self.project({"options": {}, "triangles": 40000})
        (self.root / "model.glb").write_bytes(b"glTF" + b"\0" * 40)
        (project.out / "face" / "views").mkdir(parents=True)
        (project.out / "face" / "views" / "color_torso_000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"picture")
        real = Path(report.make(self.path, note="mine", pictures=True, model=True, blender="blender")["report"])
        found = report.check(real)
        self.assertEqual((found["ok"], found["problems"]), (True, []))
        self.assertTrue(found["headline"][0].startswith("**Test** | "))

        def forged(name, **files):
            path = self.root / f"{name}.zip"
            with zipfile.ZipFile(real) as old, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as new:
                for entry in old.infolist():
                    if entry.filename not in files:
                        new.writestr(entry, old.read(entry))
                for inside, data in files.items():
                    if data is not None:
                        new.writestr(inside, data)
            return path

        def glb(doc):
            text = json.dumps({"asset": {"version": "2.0"}, **doc}).encode()
            text += b" " * (-len(text) % 4)
            return b"glTF" + struct.pack("<II", 2, 20 + len(text)) + struct.pack("<I", len(text)) + b"JSON" + text
        far = chr(92) * 2 + "elsewhere" + chr(92) + "share" + chr(92)  # a folder on another computer, as Windows writes it
        fbx = b"Kaydara FBX Binary  \x00\x1a\x00"

        cases = {
            "program": ({"runs/setup.exe": b"MZ\x90\0"}, "a .exe file"),
            "script": ({"fix.bat": b"@echo off"}, "a .bat file"),
            "renamed": ({"renders/x/sheet_face.jpg": b"MZ\x90\0 not a picture"}, "is a Windows program"),
            "not_a_picture": ({"renders/x/sheet_face.png": b"<html>"}, "not a .png file inside"),
            "blend": ({"model/source/model.glb": b"BLENDER-v402"}, "a .blend file"),
            "outside": ({"../../startup/run.json": b"{}"}, "outside the folder"),
            "drive": ({"C:/Users/x/run.json": b"{}"}, "outside the folder"),
            "stray_model": ({"runs/model.glb": b"glTF" + b"\0" * 8}, "a model outside model/"),
            "odd_folder": ({"tools/readme.txt": b"hello"}, "not a folder a report has"),
            "archive": ({"feedback/more.txt": b"PK\x03\x04"}, "is another archive"),
            "long_log": ({"runs/big.log": b"a" * (3 * 2**20)}, None),
            "no_summary": ({"report.json": None}, "no report.json"),
            "binary_text": ({"runs/x.log": b"ok\0\0\0"}, "not text inside"),
            # a model is opened by Blender, which opens what the model names: on Windows a folder on another
            # computer is signed in to with the user's own name and password hash
            "far_picture": ({"model/source/model.glb": glb({"images": [{"uri": far + "a.png"}]})}, "names a file outside the report"),
            "far_buffer": ({"model/source/model.glb": glb({"buffers": [{"uri": "https://elsewhere.example/a.bin", "byteLength": 4}]})},
                           "names a file outside the report"),
            "file_above": ({"model/source/scene.gltf": json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "../../../secret.png"}]}).encode()},
                           "names a file outside the report"),
            "far_library": ({"model/source/body.obj": b"# made by hand\nmtllib " + far.encode() + b"body.mtl\nv 0 0 0\n"}, "names a file outside the report"),
            "far_texture": ({"model/rig/rig.fbx": fbx + bytes(64) + b"\x10RelativeFilename" + b"S" + struct.pack("<I", len(far) + 8)
                             + far.encode() + b"skin.png" + bytes(32)}, "names a file outside the report"),
            "far_texture_text": ({"model/rig/rig.fbx": b'; FBX 7.4.0 project file\nTexture: 1, "Texture::skin", "" {\n\tFileName: "//elsewhere/share/skin.png"\n}\n'},
                                 "names a file outside the report"),
        }
        for name, (files, said) in cases.items():
            with self.subTest(name):
                path = forged(name, **files)
                found = report.check(path)
                if said is None:  # a spinning Blender's log is one line a million times: a log all the same
                    self.assertTrue(found["ok"], found["problems"])
                    with mock.patch.object(report, "TEXT_MOST", 2**20):
                        self.assertIn("more than such a file is in a report", " ".join(report.check(path)["problems"]))
                    continue
                self.assertFalse(found["ok"])
                self.assertIn(said, " ".join(found["problems"]))
                with self.assertRaisesRegex(ValueError, "was not unpacked"):
                    report.unpack(path, self.root / "received" / name)
                self.assertFalse((self.root / "received" / name).exists())
        self.assertFalse(report.check(self.root / "character.json")["ok"])
        # what a model may name: its own pictures inside it, and a file beside it; a texture's path on the
        # computer it was made on is in every FBX and opens nothing here
        made_on = "C:" + chr(92) + "Users" + chr(92) + "artist" + chr(92) + "skin.png"
        kept = {"model/source/model.glb": glb({"images": [{"bufferView": 0}, {"uri": "data:image/png;base64,AAAA"}, {"uri": "skin.png"}]}),
                "model/source/body.obj": b"mtllib body.mtl\nv 0 0 0\n",
                "model/rig/rig.fbx": fbx + b"\x08FileName" + b"S" + struct.pack("<I", len(made_on)) + made_on.encode()}
        self.assertEqual(report.check(forged("named_inside", **kept))["problems"], [])
        # a real one is unpacked into a folder of its own, once
        made = report.unpack(real, self.root / "received" / "real")
        self.assertEqual(json.loads((Path(made["folder"]) / "report.json").read_text())["note"], "mine")
        self.assertTrue((Path(made["folder"]) / "model" / "source" / "model.glb").is_file())
        with self.assertRaisesRegex(ValueError, "there already"):
            report.unpack(real, self.root / "received" / "real")

    def test_manifest_overrules_the_survey(self):
        project = self.project({"options": {"rigid_head": True}, "triangles": 40000}, rigid_head=False)
        self.assertFalse(project.options()["rigid_head"])
        self.assertIn("ahoge", self.runs(project))

    def test_steps_follow_the_options(self):
        project = self.project({"options": {}, "triangles": 40000}, eyes="slide", fingers="auto", tpose="none")
        runs = self.runs(project)
        self.assertIn("eyes", runs)
        self.assertNotIn("finger_guides", runs)
        self.assertNotIn("vrm_check", runs)
        self.assertNotIn("pieces", runs)  # no clothing pieces named
        # the redrawn mouth is the default since 2026-10-07 (the user's choice); null in the manifest turns it off
        self.assertIn("mouth", runs)
        self.assertNotIn("mouth", self.runs(self.project({"options": {}, "triangles": 40000}, mouth_fix=None)))

    def test_build_passes_the_manifest_its_joints_and_flags(self):
        data = manifest.load(self.path)
        data["joints"]["hips"] = [0, 0, .76]
        manifest.save(data, self.path)
        project = self.project({"options": {"rigid_head": True}, "triangles": 40000}, hat_tip=True)
        args, _ = project._arguments("build", self.root / "01_build", None)
        args = [str(a) for a in args]
        for flag in ("--manifest", "--joints", "--rigid-head", "--hat-tip", "--face", "--keep-eyes"):
            self.assertIn(flag, args)
        joints = Path(args[args.index("--joints") + 1])
        self.assertEqual(json.loads(joints.read_text()), {"hips": [0, 0, .76]})

    def test_eyes_and_mouth_arguments(self):
        project = self.project({"options": {}, "triangles": 40000}, eyes="slide", slide=[4.5, 2],
                               mouth_fix={"free": True, "scale": 1.3})
        eyes, _ = project._arguments("eyes", self.root / "10_eyes", None)
        self.assertEqual([str(a) for a in eyes[-3:]], ["--slide", "4.5", "2"])
        mouth, _ = project._arguments("mouth", self.root / "11_mouth", None)
        self.assertEqual([str(a) for a in mouth[3:]], ["--free", "--scale", "1.3"])

    def test_the_eye_choices_reach_the_eye_step(self):
        project = self.project({"options": {}, "triangles": 40000}, eyes="modelled", iris_shape="oval", iris_size=1.15,
                               look_travel=0.8, blink_amount=1.2)
        eyes, _ = project._arguments("eyes", self.root / "10_eyes", None)
        self.assertEqual([str(a) for a in eyes[-9:]],
                         ["--modelled", "--iris-shape", "oval", "--iris-size", "1.15", "--travel", "0.8", "--blink", "1.2"])
        # left as they are, none is passed; the iris's shape and size are the modelled discs' alone
        plain, _ = self.project({"options": {}, "triangles": 40000}, eyes="modelled", iris_shape=None, iris_size=1.0, look_travel=1.0,
                                blink_amount=1.0)._arguments("eyes", self.root / "10_eyes", None)
        self.assertEqual(str(plain[-1]), "--modelled")
        slide, _ = self.project({"options": {}, "triangles": 40000}, eyes="slide", slide=[4.5, 2], iris_shape="circle", iris_size=1.2,
                                look_travel=1.3)._arguments("eyes", self.root / "10_eyes", None)
        self.assertEqual([str(a) for a in slide[-5:]], ["--slide", "4.5", "2", "--travel", "1.3"])

    def test_a_covered_eye_is_told_to_both_face_steps(self):
        project = self.project({"options": {}, "triangles": 40000}, eyes="iris", eye_shown="left", iris_shape=None, iris_size=1.0,
                               look_travel=1.0, blink_amount=1.0)
        eyes, _ = project._arguments("eyes", self.root / "10_eyes", None)
        self.assertEqual([str(a) for a in eyes[-2:]], ["--only", "left"])
        face, _ = project._arguments("face_tracking", self.root / "04_face_tracking", None)
        self.assertEqual([str(a) for a in face[-2:]], ["--covered", "right"])
        # eyes behind a visor have no blink at all: nothing is said of one of them
        hidden, _ = self.project({"options": {}, "triangles": 40000}, eyes="hidden", eye_shown="left")._arguments(
            "face_tracking", self.root / "04_face_tracking", None)
        self.assertNotIn("--covered", [str(a) for a in hidden])
        self.project({"options": {}, "triangles": 40000}, eyes=None, eye_shown=None)

    def test_an_adopted_rig_keeps_its_hands_as_modelled(self):
        project = self.project({"options": {}, "triangles": 40000})
        args, _ = project._arguments("tpose", self.root / "09_tpose", None)
        self.assertIn("--spec-hands", [str(a) for a in args])
        project.data["rig"] = "rigged.fbx"
        args, _ = project._arguments("tpose", self.root / "09_tpose", None)
        self.assertNotIn("--spec-hands", [str(a) for a in args])
        self.assertIn("--keep-finger-pose", [str(a) for a in args])

    def test_hair_collides_with_the_body_until_told_otherwise(self):
        project = self.project({"options": {}, "triangles": 40000})
        args, _ = project._arguments("build", self.root / "01_build", None)
        self.assertNotIn("--hair-head-only", [str(a) for a in args])
        project = self.project({"options": {}, "triangles": 40000}, hair_hits_body=False)
        args, _ = project._arguments("build", self.root / "01_build", None)
        self.assertIn("--hair-head-only", [str(a) for a in args])

    def test_chest_bounce_is_off_until_asked_for(self):
        project = self.project({"options": {}, "triangles": 40000})
        self.assertNotIn("chest", self.runs(project))
        project = self.project({"options": {}, "triangles": 40000}, chest=True)
        self.assertIn("chest", self.runs(project))
        args, extra = project._arguments("chest", self.root / "10_chest", None)
        self.assertEqual([str(a) for a in args[:1] + args[3:]], ["add_chest_jiggle.py", "--height", "1.6", "--amount", "0.5"])
        self.assertEqual(extra, engine.NO_EXPORT)
        # on the finished T-pose, before the face steps; a chest it can't find leaves the avatar without it
        self.assertEqual(engine.NAMES.index("chest"), engine.NAMES.index("tpose") + 1)
        self.assertIn("chest", engine.SOFT)

    def test_an_iris_the_finished_file_shows_a_socket_through_falls_back_to_the_discs(self):
        # the fox girl's fresh build: the moving iris passed, the drawn file showed the inside of her head
        project = self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}})
        self.assertEqual(project.options()["eyes"], "iris")
        project.state["steps"]["vrm_check"] = {"status": "done", "report": {"seen": {"failed": ["102.5 mm2 seen deep"]}}}
        self.assertTrue(project._iris_seen_through())
        project._seen_retry = True  # once only
        self.assertFalse(project._iris_seen_through())
        project._seen_retry = False
        project.state["steps"]["vrm_check"]["report"]["seen"] = {"failed": []}
        self.assertFalse(project._iris_seen_through())
        # a file that wasn't drawn isn't taken for a hole: said instead, and the build isn't ready (verdict)
        project.state["steps"]["vrm_check"]["report"]["seen"] = {"left": "the file couldn't be drawn", "code": "VM-NOT-DRAWN"}
        self.assertFalse(project._iris_seen_through())
        # nor an iris the user chose, nor a model with no discs to fall back to
        project.state["steps"]["vrm_check"]["report"]["seen"] = {"failed": ["102.5 mm2 seen deep"]}
        self.assertFalse(self.project({"options": {}, "triangles": 40000}, eyes="iris")._iris_seen_through())
        self.assertFalse(self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": False}})._iris_seen_through())

    def test_a_refused_automatic_iris_slides_instead(self):
        project = self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}})
        self.assertEqual(project.options()["eyes"], "iris")
        refused = engine.StepFailed("eyes: iris over the lids", errors.describe("VM-EYES-REFUSED"))
        # first the model's own iris discs, then the painted patch, then the blink alone, then nothing more
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "modelled")
        args, _ = project._arguments("eyes", self.root / "11_eyes", None)
        self.assertIn("--modelled", [str(a) for a in args])
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "slide")
        args, _ = project._arguments("eyes", self.root / "11_eyes", None)
        self.assertIn("--slide", [str(a) for a in args])
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "blink")
        args, _ = project._arguments("eyes", self.root / "11_eyes", None)
        self.assertIn("--blink-only", [str(a) for a in args])
        self.assertFalse(project._eyes_fall_back("eyes", refused))
        # kept for the next build of the same model
        self.assertEqual(Project(self.path, blender="blender").options()["eyes"], "blink")
        # an iris the user chose stays an iris, and other failures aren't retried
        chosen = self.project({"options": {}, "triangles": 40000}, eyes="iris")
        self.assertFalse(chosen._eyes_fall_back("eyes", refused))
        self.assertFalse(project._eyes_fall_back("mouth", refused))

    def test_a_remembered_patch_that_is_refused_leaves_the_iris_to_try(self):
        # a tester's fox girl: "slide" kept from a build of before her iris discs could slide, refused on
        # every build since, and nothing else was tried
        project = self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}}, eyes=None)
        project.state["eyes_fallback"] = "slide"
        self.assertEqual(project.options()["eyes"], "slide")
        refused = engine.StepFailed("eyes: face over the eye patch", errors.describe("VM-EYES-REFUSED"))
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "iris")
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "modelled")
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "blink")
        self.assertFalse(project._eyes_fall_back("eyes", refused))
        # painted eyes slide from the start: a refusal there leaves the blink alone (the chibi's lids
        # had only the face step's blink, sized from her inserts)
        painted = self.project({"options": {}, "triangles": 40000, "parts": {}}, eyes=None)
        painted.data["face"] = "face.json"
        painted.state.pop("eyes_fallback", None)  # (the same state file as the project above: a model of its own)
        self.assertEqual(painted.options()["eyes"], "slide")
        self.assertTrue(painted._eyes_fall_back("eyes", refused))
        self.assertEqual(painted.options()["eyes"], "blink")
        self.assertFalse(painted._eyes_fall_back("eyes", refused))

    def test_a_refused_patch_is_not_told_to_slide(self):
        known = errors.describe("VM-EYES-REFUSED")
        refused = engine.StepFailed(f"eyes: failed [VM-EYES-REFUSED] {known['fix']}", known)
        modelled = self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}}, eyes="slide")
        advice = modelled._eyes_advice("eyes", refused)
        self.assertEqual(advice.known["options"], {"eyes": "modelled"})
        self.assertNotIn("set eyes to slide", str(advice))
        painted = self.project({"options": {}, "triangles": 40000, "parts": {}}, eyes="slide")
        self.assertIsNone(painted._eyes_advice("eyes", refused).known["options"])
        # an iris refused is still told to slide
        iris = self.project({"options": {}, "triangles": 40000}, eyes="iris")
        self.assertIs(iris._eyes_advice("eyes", refused), refused)

    def test_a_listing_made_before_the_eyes_were_marked_is_stale(self):
        project = self.project({"options": {}, "triangles": 40000, "face_patch": True})
        self.assertFalse(cli.listing_stale(project, {"pieces": []}))  # no face marked
        (self.root / "face.json").write_text(json.dumps({"size": [8, 8], "left_eye": {}, "right_eye": {"inner": [1, 1]}}))
        project.data["face"] = str(self.root / "face.json")
        self.assertTrue(cli.listing_stale(project, {"pieces": []}))
        self.assertFalse(cli.listing_stale(project, {"pieces": [], "eyes_marked": True}))
        self.assertFalse(cli.listing_stale(project, None))

    def test_only_the_export_step_writes_a_vrm(self):
        project = self.project({"options": {}, "triangles": 40000}, tpose="none")
        kinds = {name: kind for name, _, kind in engine.STEPS}
        self.assertEqual([name for name in engine.NAMES if kinds[name] == "vrm"], ["export"])
        for name in ("build", "hair", "tpose", "eyes"):
            _, extra = project._arguments(name, self.root / name, None)
            self.assertEqual(extra, engine.NO_EXPORT, name)
        args, extra = project._arguments("export", self.root / "13_export", None)
        self.assertEqual(([str(a) for a in args], extra), (["export_vrm.py", str(self.root / "13_export.vrm")], []))
        # after the last step that changes the model, before the check that re-imports the file
        self.assertLess(engine.NAMES.index("mouth"), engine.NAMES.index("export"))
        self.assertLess(engine.NAMES.index("export"), engine.NAMES.index("vrm_check"))
        self.assertIn("export", self.runs(project))  # a build kept in its modelled pose can still be looked at

    def test_a_rerun_step_makes_later_ones_stale_and_they_are_not_read(self):
        project = self.project({"options": {}, "triangles": 40000})
        files = {}
        for name in ("build", "hair", "tpose"):
            files[name] = self.root / f"{name}.blend"
            files[name].write_text("")
            project._record(name, {"status": "done", "blend": str(files[name])})
        self.assertEqual(project.latest(), files["tpose"])
        project._record("hair", {"status": "done", "blend": str(files["hair"])})
        self.assertEqual(project.state["steps"]["tpose"]["status"], "stale")
        self.assertEqual(project.latest(), files["hair"])
        self.assertEqual(project.latest(before="hair"), files["build"])

    def test_only_one_blender_at_a_time(self):
        real, engine.LOCK = engine.LOCK, self.root / ".blender.lock"
        try:
            with engine._Lock():
                with self.assertRaises(engine.StepFailed):
                    with engine._Lock():
                        pass
            self.assertFalse(engine.LOCK.exists())
        finally:
            engine.LOCK = real

    def test_stop_kills_the_running_blender_and_frees_the_lock(self):
        killed = []

        class Hanging:
            returncode = None

            def __init__(self, *args, **kwargs):
                pass

            def poll(self):
                return -9 if killed else None

            def kill(self):
                killed.append(True)

            def wait(self):
                return -9
        project = self.project()
        real = engine.LOCK, engine.subprocess.Popen, engine.time.sleep
        engine.LOCK, engine.subprocess.Popen, engine.time.sleep = self.root / ".blender.lock", Hanging, lambda s: engine.CANCEL.set()
        try:
            with self.assertRaises(engine.Cancelled):
                project.run(only="survey")
            self.assertTrue(killed)
            # what the step last finished stays: a stopped rerun must not lose it
            self.assertNotIn("survey", project.state["steps"])
            self.assertFalse(engine.LOCK.exists())
            # a stop between steps: the next one doesn't start
            with self.assertRaisesRegex(engine.Cancelled, "before survey"):
                project.run(only="survey")
        finally:
            engine.LOCK, engine.subprocess.Popen, engine.time.sleep = real
            engine.CANCEL.clear()

    def test_runs_are_listed_newest_first_with_their_vrm_and_check(self):
        project = self.project()
        for run, files in (("20260101-100000", ["01_build.log", "01_build.vrm"]),
                           ("20260102-100000", ["09_tpose.log", "09_tpose.vrm", "12_check.log", "12_check.json"])):
            folder = project.out / "runs" / run
            folder.mkdir(parents=True)
            for name in files:
                (folder / name).write_text(json.dumps({"failed": ["an edge stretches"]}) if name.endswith(".json") else "")
        project.state["steps"]["tpose"] = {"status": "done", "vrm": str(project.out / "runs" / "20260102-100000" / "09_tpose.vrm")}
        runs = project.runs()
        self.assertEqual([r["run"] for r in runs], ["20260102-100000", "20260101-100000"])
        self.assertEqual(runs[0]["steps"], ["tpose", "check"])
        self.assertEqual(Path(runs[0]["vrm"]).name, "09_tpose.vrm")
        self.assertEqual(runs[0]["failed"], ["an edge stretches"])
        self.assertEqual([r["current"] for r in runs], [True, False])
        self.assertIsNone(runs[1]["check"])

    def test_a_character_card_says_how_far_the_build_got(self):
        project = self.project()
        self.assertEqual(project.summary(), {"surveyed": False, "built": False, "stale": [], "failed": [],
                                             "finished": None, "approved": False, "picture": None, "portrait": None,
                                             "stills": {}, "portraits": {}, "waiting": False})
        self.finished(project, eyes={"status": "failed", "finished": "2026-01-02T10:00:00"}, tpose={"status": "stale"})
        pictures = project.out / "renders" / "20260102-100000_09_tpose_raised"
        pictures.mkdir(parents=True)
        for view in ("back", "front"):
            (pictures / f"raised_{view}.png").write_bytes(b"")
        summary = project.summary()
        self.assertTrue(summary["built"])
        self.assertEqual((summary["failed"], summary["stale"]), (["eyes"], ["tpose"]))
        self.assertEqual(Path(summary["picture"]).name, "raised_front.png")
        # the portrait: a front render with the arms down if there is one, and never a contact sheet, however new
        self.assertEqual(Path(summary["portrait"]).name, "raised_front.png")
        for folder, name in (("20260101-090000_09_tpose_hang", "hang_front.png"), ("sheets_20260103-100000", "sheet_front.png")):
            (project.out / "renders" / folder).mkdir()
            (project.out / "renders" / folder / name).write_bytes(b"")
        self.assertEqual(Path(project.summary()["portrait"]).name, "hang_front.png")
        self.assertEqual(sorted(project.summary()["stills"]), ["hang_front", "raised_back", "raised_front"])
        # a T-pose is the rig's rest, never the character's picture
        (project.out / "renders" / "20260101-090000_09_tpose_hang" / "hang_front.png").unlink()
        (project.out / "renders" / "20260104-090000_09_tpose_t-pose").mkdir()
        (project.out / "renders" / "20260104-090000_09_tpose_t-pose" / "t-pose_front.png").write_bytes(b"")
        self.assertEqual(Path(project.summary()["portrait"]).name, "raised_front.png")
        # the posed pictures, once drawn: each with where its head and floor are
        posed = project.out / "portraits"
        posed.mkdir()
        (posed / "stand.png").write_bytes(b"")
        (posed / "portraits.json").write_text(json.dumps({"blend": "x.blend", "height": 0.64, "face": 0.1, "poses": {
            "stand": {"file": "stand.png", "head": [0.5, 0.3], "floor": 0.9}, "wave": {"file": "wave.png", "head": [0.5, 0.3], "floor": 0.9}}}))
        self.assertEqual(project.summary()["portraits"], {"stand": {"file": str(posed / "stand.png"), "head": [0.5, 0.3], "floor": 0.9, "height": 0.64, "face": 0.1}})

    def face_points(self):
        eye = {"inner": [10, 20], "outer": [30, 20], "top": [20, 15], "bottom": [20, 25]}
        return {"image": "face_patch.png", "size": [100, 100], "right_eye": eye, "left_eye": eye,
                "mouth": {"left": [40, 60], "right": [60, 60], "top": [50, 58], "bottom": [50, 62]}}

    def test_face_points_are_kept_with_the_character(self):
        saved = cli.save_face(self.path, self.face_points())
        self.assertEqual(Path(saved["face"]), (self.root / "out" / "face.json").resolve())
        self.assertEqual(manifest.path_of(manifest.load(self.path), "face"), Path(saved["face"]))
        self.assertEqual(json.loads(Path(saved["face"]).read_text())["mouth"]["left"], [40, 60])

    def test_face_points_missing_one_are_refused(self):
        points = self.face_points()
        del points["mouth"]["bottom"]
        with self.assertRaisesRegex(ValueError, "mouth is missing bottom"):
            cli.save_face(self.path, points)

    def test_a_face_with_covered_eyes_is_its_mouth_alone(self):
        # a visor or glasses over the eyes: nothing to mark there, and nothing to blink or move
        points = {**self.face_points(), "right_eye": {}, "left_eye": {}}
        saved = cli.save_face(self.path, points)
        self.assertEqual(set(json.loads(Path(saved["face"]).read_text())), {"image", "size", "mouth"})
        project = self.project({"options": {}, "triangles": 40000, "face_patch": True}, eyes="iris")
        self.assertEqual(project.options()["eyes"], "hidden")
        self.assertIn("face_tracking", self.runs(project))
        self.assertNotIn("eyes", self.runs(project))
        del points["mouth"]["top"]
        with self.assertRaisesRegex(ValueError, "mouth is missing top"):
            cli.save_face(self.path, points)

    def test_one_eye_marked_is_still_refused(self):
        points = {**self.face_points(), "left_eye": {}}
        with self.assertRaisesRegex(ValueError, "left_eye is missing"):
            cli.save_face(self.path, points)

    def finished(self, project, **steps):
        """Records the build as run to the end, plus whatever other step entries a test gives."""
        project.state["steps"]["vrm_check"] = {"status": "done"}
        for name, entry in steps.items():
            project.state["steps"][name] = entry

    def test_verdict_waits_for_the_end_of_the_build(self):
        verdict = self.project({"options": {}, "triangles": 40000}).verdict()
        self.assertFalse(verdict["ready"])
        self.assertEqual(verdict["issues"], [("build", "not built to the end yet")])

    def test_verdict_blocks_on_the_rig_check_and_a_missing_feature(self):
        project = self.project({"options": {}, "triangles": 40000})
        self.finished(project, eyes={"status": "failed"})
        verdict = project.verdict(["arms raised: stretch"])
        self.assertFalse(verdict["ready"])
        self.assertIn(("review", "rig check: arms raised: stretch"), verdict["issues"])
        self.assertIn(("build", "built without eyes"), verdict["issues"])

    def test_movement_and_spring_findings_are_looks_not_blockers(self):
        project = self.project({"options": {}, "triangles": 40000})
        springs = self.root / "springs.json"
        springs.write_text(json.dumps({"sagging": [{"chain": "skirt_0", "sag_cm": 5.0}],
                                       "coming_away": [{"count": 12, "gap_cm": 14.6}]}))
        self.finished(project, springs={"status": "done", "json": str(springs)}, movement={"status": "failed"})
        verdict = project.verdict([])
        self.assertTrue(verdict["ready"])
        self.assertEqual(verdict["issues"], [])
        self.assertEqual(verdict["looks"], [("movement", "skirt_0 droops 5.0 cm standing still"),
                                            ("movement", "a piece of 12 vertices is left 14.6 cm behind what it sits on"),
                                            ("movement", "the movement check couldn't run")])

    def test_a_build_that_found_no_hair_says_so(self):
        # the fox girl's first build: "ready", and her hair on no spring
        said = ("parts", "no hair was found to hang on springs, so none swings: right for short hair, a hat or a "
                         "hood. If this character's hair should move, paint it on the Parts page and call it hair")
        project = self.project({"options": {}, "triangles": 40000})
        self.finished(project, build={"status": "done", "report": {"hair_chains": 0, "cloth_chains": 3}})
        verdict = project.verdict([])
        self.assertTrue(verdict["ready"])
        self.assertIn(said, verdict["looks"])
        self.finished(project, build={"status": "done", "report": {"hair_chains": 12, "cloth_chains": 3}})
        self.assertNotIn(said, project.verdict([])["looks"])
        # a head asked to be rigid has no hair to find; one the survey took for rigid may have been taken wrongly
        rigid = self.project({"options": {}, "triangles": 40000}, rigid_head=True)
        self.finished(rigid, build={"status": "done", "report": {"hair_chains": 0}})
        self.assertNotIn(said, rigid.verdict([])["looks"])

    def test_a_tidy_up_undone_for_tearing_the_rig_is_worth_a_look(self):
        project = self.project({"options": {}, "triangles": 40000})
        undone = "the tidy-up tore the rig (arms raised 3.5 -> 77.6 cm): done again without joining touching surfaces"
        self.finished(project, loose={"status": "done", "report": {"stretch_cm": {"arms raised": 3.5}, "undone": [undone]}})
        verdict = project.verdict([])
        self.assertTrue(verdict["ready"])
        self.assertIn(("build", f"loose pieces: {undone}"), verdict["looks"])

    def test_a_rebuild_is_compared_with_the_build_marked_good(self):
        project = self.project({"options": {}, "triangles": 40000})

        def report(name, data):
            path = self.root / f"{name}_{len(list(self.root.iterdir()))}.json"
            path.write_text(json.dumps(data))
            project.state["steps"][name] = {"status": "done", "json": str(path)}

        check = lambda head, arms: {"poses": {"head turn": {"stretch_cm": head}, "arms raised": {"stretch_cm": arms, "squash": .6}},
                                    "unweighted": 0, "worst": {}, "failed": []}
        report("check", check(2.6, 5.0))
        report("vrm_check", check(2.6, 5.0))
        report("movement", {"pieces": []})
        report("springs", {"sagging": [{"chain": "skirt_0", "sag_cm": 2.0}], "coming_away": []})
        self.assertIsNone(project.since_approved())
        project.approve("looked right in Warudo")
        self.assertEqual(project.since_approved(), {"worse": [], "better": []})
        report("check", check(8.0, 4.0))
        report("movement", {"pieces": [{"label": 3, "name": "twintail_R", "findings": [{"kind": "torn", "pose": "arms raised"}]}]})
        report("springs", {"sagging": [{"chain": "skirt_0", "sag_cm": 5.0}], "coming_away": []})
        changed = project.since_approved()
        self.assertEqual(changed["worse"], ["rig check, head turn: stretch 2.6 -> 8.0 cm", "movement: twintail_R newly flagged",
                                            "physics: worst droop at rest 2.0 -> 5.0 cm"])
        self.assertEqual(changed["better"], ["rig check, arms raised: stretch 5.0 -> 4.0 cm"])
        self.assertIn(("build", "worse than the approved build: rig check, head turn: stretch 2.6 -> 8.0 cm"),
                      project.verdict([])["looks"])

    def test_droop_is_compared_chain_by_chain(self):
        # only chains over their own limit are listed: a hat tip's 4 cm against a cape's 8.6 is not one chain drooping more
        def report(name, sagging):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps({"sagging": [{"chain": c, "sag_cm": v} for c, v in sagging.items()], "coming_away": []}))
            return {"springs": path}

        before = report("before", {"HatTip": 4.0, "Skirt": 9.0})
        now = report("now", {"HatTip": 3.5, "Cloth9": 8.6, "Cloth10": 8.5})
        self.assertEqual(engine.changes(before, now), {
            "worse": ["physics: 2 chains newly over the droop limit at rest (Cloth9 8.6 cm)"],
            "better": ["physics: 1 chain no longer over the droop limit at rest (Skirt was 9.0 cm)"]})
        # the chain that droops further is the one named, though another droops more
        self.assertEqual(engine.changes(before, report("further", {"HatTip": 6.0, "Skirt": 9.2})),
                         {"worse": ["physics: worst droop at rest 4.0 -> 6.0 cm"], "better": []})

    def test_pieces_left_behind_are_compared_one_by_one(self):
        # model A: a bit of hair had stood still since September (29.8 cm); chest panels newly turning
        # with the head (22.5, 14.4 cm) were under that worst gap, and regress said "nothing worse"
        def report(name, pieces):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps({"sagging": [], "coming_away": [
                {"label": i, "count": count, "gap_cm": gap, "point": point} for i, (count, gap, point) in enumerate(pieces)]}))
            return {"springs": path}

        hair = (180, 29.8, [0.224, 0.133, 1.023])
        before = report("before", [hair])
        now = report("now", [hair, (315, 22.5, [-0.107, -0.183, 1.111]), (280, 14.4, [-0.05, -0.144, 1.19])])
        self.assertEqual(engine.changes(before, now),
                         {"worse": ["physics: 2 pieces newly left behind by a swing (315 vertices, 22.5 cm)"], "better": []})
        self.assertEqual(engine.changes(now, before),
                         {"worse": [], "better": ["physics: 2 pieces no longer left behind by a swing (315 vertices, 22.5 cm)"]})
        # the same piece a rebuild numbered otherwise and found a few millimetres off is no news
        moved = report("moved", [(181, 29.6, [0.226, 0.131, 1.02])])
        self.assertEqual(engine.changes(before, moved), {"worse": [], "better": []})

    def test_face_shapes_are_compared_key_by_key(self):
        # model D: mouth points saved again on her throat, and two days of regress said "nothing worse"
        def report(name, keys):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps({"poses": {}, "unweighted": 0, "face": {"keys": keys}}))
            return {"check": path}

        key = lambda moved, reach, z: {"moved": moved, "max_mm": reach, "at": [0.0, -0.27, z]}  # noqa: E731
        good = report("good", {"aa": key(98, 17.31, 1.338), "jawOpen": key(1116, 18.52, 1.322), "blink": key(300, 6.0, 1.4)})
        self.assertEqual(engine.changes(good, report("again", {"aa": key(98, 17.3, 1.339), "jawOpen": key(1110, 18.6, 1.322),
                                                                "blink": key(300, 6.0, 1.4)})), {"worse": [], "better": []})
        throat = report("throat", {"aa": key(99, 9.16, 1.279), "jawOpen": key(133, 10.78, 1.28), "blink": key(300, 6.0, 1.4)})
        self.assertEqual(engine.changes(good, throat)["worse"],
                         ["face shapes: 2 of 3 changed, aa works 5.9 cm from where it did"])
        emptied = report("emptied", {"aa": key(98, 17.31, 1.338), "jawOpen": {"moved": 0, "max_mm": 0.0}})
        self.assertEqual(engine.changes(good, emptied)["worse"],
                         ["face shapes: 2 of 3 changed, jawOpen no longer moves anything (it moved 1116 vertices)"])

    def test_a_build_older_than_the_face_part_of_the_check_is_read_from_its_last_fingerprint(self):
        # no places in those: the count and the reach alone still told model D's throat from her face
        for name, jaw in (("old", {"moved": 1116, "max_mm": 18.52}), ("new", {"moved": 133, "max_mm": 10.78})):
            (self.root / name).mkdir()
            (self.root / name / "13_check.json").write_text(json.dumps({"poses": {}, "unweighted": 0}))
            (self.root / name / "01_build_fingerprint.json").write_text(json.dumps({"keys": {}}))
            (self.root / name / "12_mouth_fingerprint.json").write_text(json.dumps({"keys": {"body": {"jawOpen": jaw}}}))
        old, new = ({"check": self.root / name / "13_check.json"} for name in ("old", "new"))
        self.assertEqual(engine.changes(old, new)["worse"],
                         ["face shapes: 1 of 1 changed, jawOpen moves 133 vertices by up to 10.78 mm, was 1116 by 18.52"])
        self.assertEqual(engine.changes(old, old), {"worse": [], "better": []})

    def test_only_a_finished_build_can_be_marked_good(self):
        with self.assertRaisesRegex(engine.StepFailed, "build to the end"):
            self.project({"options": {}, "triangles": 40000}).approve()

    def test_no_face_points_is_said_on_the_verdict(self):
        project = self.project({"options": {}, "triangles": 40000})
        project.data["face"] = None
        self.finished(project)
        self.assertEqual(project.verdict([])["looks"][0][0], "face")
        project.data["face"] = "face.json"
        self.assertFalse(any(page == "face" for page, _ in project.verdict([])["looks"]))

    def test_renaming_a_piece_carries_the_details_anchored_to_it(self):
        data = manifest.load(self.path)
        data["pieces"] = [{"name": "appendage_38", "point": [0, 0, 1], "count": 5, "role": "appendage"},
                          {"name": "trim", "point": [0, 0, 1.1], "count": 2, "role": "detail", "anchor": "appendage_38"}]
        manifest.save(data, self.path)
        args = lambda name, rename: type("A", (), {"manifest": self.path, "name": name, "rename": rename})()
        cli.cmd_piece(args("appendage_38", "left wing"))
        pieces = manifest.load(self.path)["pieces"]
        self.assertEqual([p["name"] for p in pieces], ["left wing", "trim"])
        self.assertEqual(pieces[1]["anchor"], "left wing")
        with self.assertRaisesRegex(ValueError, "already named"):
            cli.cmd_piece(args("trim", "left wing"))

    def test_a_piece_given_another_role_leaves_what_the_old_role_was_told(self):
        # a hanging sleeve's arm bone, kept when the piece was called a tail instead, hung the tail from the arm
        data = manifest.load(self.path)
        data["pieces"] = [{"name": "tail", "point": [0, 0, 1], "count": 5, "role": "follow_bone", "bone": "LeftUpperArm"},
                          {"name": "hat", "point": [0, 0, 1.5], "count": 9, "role": "hat", "tip": True}]
        manifest.save(data, self.path)
        unsaid = {"role": None, "point": None, "count": None, "label": None, "bone": None, "anchor": None, "tip": None,
                  "remove": False, "region": None, "rename": None}
        piece = lambda name, **said: cli.cmd_piece(SimpleNamespace(manifest=self.path, name=name, **{**unsaid, **said}))["piece"]  # noqa: E731
        self.assertEqual(piece("tail", role="appendage"), {"name": "tail", "point": [0, 0, 1], "count": 5, "role": "appendage"})
        self.assertEqual(piece("tail", role="appendage", bone="Hips")["bone"], "Hips")
        self.assertEqual(piece("tail", role="appendage")["bone"], "Hips")  # the same role again: what it was told stays
        self.assertEqual(piece("tail", role="follow_bone", bone="RightUpperArm")["bone"], "RightUpperArm")
        # "tip swings" could be ticked on the Parts page and never unticked
        self.assertTrue(piece("hat", role="hat")["tip"])
        self.assertNotIn("tip", piece("hat", role="hat", tip=False))
        self.assertTrue(piece("hat", role="hat", tip=True)["tip"])
        self.assertNotIn("tip", piece("hat", role="hair"))
        # ... and as the page says it: false takes the tip off, unsaid leaves it
        server.api("piece", {}, {"manifest": str(self.path), "name": "hat", "role": "hat", "tip": True})
        self.assertTrue(server.api("piece", {}, {"manifest": str(self.path), "name": "hat", "role": "hat"})["piece"]["tip"])
        self.assertNotIn("tip", server.api("piece", {}, {"manifest": str(self.path), "name": "hat", "role": "hat", "tip": False})["piece"])

    def test_warudo_feedback_is_kept_with_its_screenshots(self):
        import base64
        png = "data:image/png;base64," + base64.b64encode(bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]) + b"0" * 20).decode()
        saved = cli.save_feedback(self.path, "left sleeve pokes through", [png])
        self.assertEqual(saved["screenshots"], ["screenshot_1.png"])
        self.assertTrue((Path(saved["folder"]) / "screenshot_1.png").exists())
        self.assertEqual(cli.list_feedback(self.path)[0]["note"], "left sleeve pokes through")
        with self.assertRaisesRegex(ValueError, "PNG or JPEG"):
            cli.save_feedback(self.path, "x", ["data:text/html;base64,PGI+"])
        with self.assertRaisesRegex(ValueError, "say what you saw"):
            cli.save_feedback(self.path, "  ")

    def test_a_note_keeps_its_ticks_and_verdict_apart(self):
        saved = cli.save_feedback(self.path, "", checks=["arms up: sleeves and shoulders follow"], verdict="right")
        self.assertEqual((saved["note"], saved["checks"], saved["verdict"]), ("", ["arms up: sleeves and shoulders follow"], "right"))
        with self.assertRaisesRegex(ValueError, "verdict"):
            cli.save_feedback(self.path, "x", verdict="perfect")
        # a note from before they were kept apart still lists
        old = Project(self.path).out / "feedback" / "20260101-100000"
        old.mkdir(parents=True)
        (old / "feedback.json").write_text(json.dumps({"when": "2026-01-01T10:00:00", "note": "old", "vrm": "", "screenshots": []}))
        self.assertEqual([(f["note"], f["checks"], f["verdict"]) for f in cli.list_feedback(self.path)],
                         [("", ["arms up: sleeves and shoulders follow"], "right"), ("old", [], None)])

    def test_a_finished_build_waits_until_someone_has_looked(self):
        project = self.project()
        self.assertIsNone(project.looked_at())  # nothing built
        run = project.out / "runs" / "20260102-100000"
        run.mkdir(parents=True)
        (run / "14_export.vrm").write_bytes(b"vrm")
        self.finished(project, export={"status": "done", "vrm": str(run / "14_export.vrm")})
        project._save_state()
        self.assertIs(project.looked_at(), False)
        self.assertTrue(project.summary()["waiting"])
        cli.save_feedback(self.path, "blink fine")
        self.assertIs(Project(self.path).looked_at(), True)
        # a later build is a new file: the note was about the old one
        later = project.out / "runs" / "20260103-100000"
        later.mkdir(parents=True)
        (later / "14_export.vrm").write_bytes(b"vrm")
        project.state["steps"]["export"]["vrm"] = str(later / "14_export.vrm")
        self.assertIs(project.looked_at(), False)

    def test_compare_rows_say_which_way_each_pose_moved(self):
        before, after = self.root / "before.json", self.root / "after.json"
        before.write_text(json.dumps({"unweighted": 0, "poses": {"arms raised": {"stretch_cm": 5.0, "squash": 0.5}, "head turn": {"stretch_cm": 3.0, "squash": 0.6}}}))
        after.write_text(json.dumps({"unweighted": 2, "poses": {"arms raised": {"stretch_cm": 9.0, "squash": 0.5}, "head turn": {"stretch_cm": 3.1, "squash": 0.8}}}))
        rows = cli.compare_rows(before, after)
        self.assertEqual(rows["unweighted"], [0, 2])
        self.assertEqual([(r["pose"], r["stretch"], r["stretch_way"], r["squash_way"]) for r in rows["poses"]],
                         [("arms raised", [5.0, 9.0], "worse", ""), ("head turn", [3.0, 3.1], "", "better")])
        self.assertEqual(cli.compare_rows(before, self.root / "missing.json"), {"poses": [], "unweighted": None})

    def test_a_character_is_renamed_where_it_stands_and_hidden_without_deleting(self):
        project = self.project()
        out = project.out
        renamed = cli.rename(self.path, "Other")
        self.assertEqual((renamed["name"], renamed["was"]), ("Other", "Test"))
        again = Project(self.path, blender="blender")
        self.assertEqual((again.name, again.out), ("Other", out))  # the builds stay where they are
        for bad in ("../x", "a b", ""):
            with self.assertRaisesRegex(ValueError, "folder name"):
                cli.rename(self.path, bad)
        real = engine.REPO
        engine.REPO = self.root
        try:
            (self.root / "out" / "Other").mkdir(parents=True)
            listed = self.root / "out" / "Other" / "character.json"
            manifest.save(manifest.load(self.path), listed)
            self.assertEqual(engine.characters(), [listed])
            engine.hide_character(listed)
            self.assertEqual((engine.characters(), engine.hidden_characters()), ([], [str(listed.resolve())]))
            self.assertTrue(listed.exists())
            engine.hide_character(listed, hide=False)
            self.assertEqual(engine.characters(), [listed])
        finally:
            engine.REPO = real

    def test_deleting_a_character_takes_its_folder_and_leaves_what_others_read(self):
        (self.root / "model.glb").write_bytes(b"glb")
        real = engine.REPO, cli.REPO, engine.LOCK
        engine.REPO, cli.REPO, engine.LOCK = self.root, self.root, self.root / "lock"
        try:
            first = Path(cli.duplicate(self.path, "First")["manifest"])
            patched = first.parent / "face" / "First_face.glb"
            patched.parent.mkdir()
            patched.write_bytes(b"patched")
            data = manifest.load(first)
            data["source"], data["reference"] = "face/First_face.glb", "../../model.glb"
            manifest.save(data, first)
            copy = Path(cli.duplicate(first, "Copy")["manifest"])
            # the copy is built from the first one's face patch
            with self.assertRaisesRegex(ValueError, "Copy is built from files in .*First"):
                engine.delete_character(first)
            self.assertTrue(patched.exists())
            would = engine.delete_character(copy, dry=True)
            self.assertTrue(copy.exists())
            self.assertEqual((would["folder"], would["model"], would["dry"]), (str(copy.parent), str(self.root / "model.glb"), True))
            engine.delete_character(copy)
            self.assertFalse(copy.parent.exists())
            gone = engine.delete_character(first)
            self.assertFalse(first.parent.exists())
            self.assertGreater(gone["freed"], len(b"patched"))
            self.assertEqual(engine.characters(), [])
            # an output folder named by hand, anywhere on the disk, is not the app's to delete
            kept = self.root / "out" / "old.vrm"
            kept.write_bytes(b"vrm")
            done = engine.delete_character(self.path)
            self.assertEqual((done["folder"], done["left"]), (None, str(self.root / "out")))
            self.assertTrue(kept.exists() and (self.root / "model.glb").exists())
            self.assertFalse(self.path.exists())
            with self.assertRaisesRegex(ValueError, "isn't a character"):
                engine.delete_character(self.root / "model.glb")
        finally:
            engine.REPO, cli.REPO, engine.LOCK = real

    def test_a_path_saved_with_its_quotes_reads_as_the_path_meant(self):
        # what `init` wrote for a path pasted from Explorer's "Copy as path", before it took the quotes off
        for saved, meant in (('..\\..\\"C:\\Users\\bob\\my oc.fbx"', "C:\\Users\\bob\\my oc.fbx"), ('"C:\\a\\b.fbx', "C:\\a\\b.fbx"),
                             ("..\\..\\'D:/a/b.glb'", "D:/a/b.glb"), ("/home/bob/bob's oc.fbx", "/home/bob/bob's oc.fbx"),
                             ("../model.glb", "../model.glb")):
            data = json.loads(self.path.read_text())
            data["source"] = data["rig"] = saved
            self.path.write_text(json.dumps(data))
            read = manifest.load(self.path)
            self.assertEqual((read["source"], read["rig"]), (meant, meant))

    def test_a_duplicate_is_the_same_character_with_its_own_builds(self):
        (self.root / "face.json").write_text(json.dumps({"size": [8, 8]}))
        data = manifest.load(self.path)
        data["options"]["eyes"] = "iris"
        data["joints"] = {"hips": [0.0, 0.0, 0.9]}
        manifest.save(data, self.path)
        (self.root / "out").mkdir()
        (self.root / "out" / "pieces.json").write_text("{}")
        real = cli.REPO
        cli.REPO = self.root
        try:
            made = cli.duplicate(self.path, "Test_b")
            with self.assertRaisesRegex(ValueError, "exists already"):
                cli.duplicate(self.path, "Test_b")
            for name in ("../evil", "a b", ""):
                with self.assertRaisesRegex(ValueError, "folder name"):
                    cli.duplicate(self.path, name)
        finally:
            cli.REPO = real
        copy = manifest.load(made["manifest"])
        self.assertEqual(Path(made["manifest"]), self.root / "out" / "Test_b" / "character.json")
        self.assertEqual((copy["name"], copy["out_dir"], copy["options"]["eyes"], copy["joints"]), ("Test_b", None, "iris", {"hips": [0.0, 0.0, 0.9]}))
        self.assertEqual(manifest.path_of(copy, "source"), manifest.path_of(manifest.load(self.path), "source"))
        self.assertEqual(manifest.path_of(copy, "face"), (self.root / "out" / "Test_b" / "face.json").resolve())
        self.assertTrue((self.root / "out" / "Test_b" / "pieces.json").exists())
        self.assertFalse((self.root / "out" / "Test_b" / "state.json").exists())


    def test_a_half_written_state_falls_back_to_its_backup(self):
        project = self.project()
        project._record("build", {"status": "done", "blend": "a.blend"})
        project._record("hair", {"status": "done", "blend": "b.blend"})
        project.state_path.write_text('{"steps": {"bui', encoding="utf-8")  # a crash mid-write
        self.assertEqual(Project(self.path).state["steps"]["build"]["blend"], "a.blend")
        project.state_path.with_name("state.json.bak").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(engine.StepFailed, "no usable backup"):
            Project(self.path)

    def test_a_lock_let_go_while_it_was_looked_at_is_taken(self):
        real, engine.LOCK = engine.LOCK, self.root / ".blender.lock"
        opened, tries = engine.os.open, []

        def held_once(*args):  # the other build lets go between the refused try and the look at who holds it
            tries.append(args)
            if len(tries) == 1:
                raise FileExistsError
            return opened(*args)
        engine.os.open = held_once
        try:
            with engine._Lock():
                self.assertIn(f"pid {engine.os.getpid()}", engine.LOCK.read_text())
            self.assertEqual(len(tries), 2)
        finally:
            engine.os.open, engine.LOCK = opened, real

    def test_a_run_refused_for_another_build_leaves_no_run_folder(self):
        project = self.project()
        real, engine.LOCK = engine.LOCK, self.root / ".blender.lock"
        try:
            with engine._Lock():
                with self.assertRaisesRegex(engine.StepFailed, "another build is running"):
                    project.run(only="survey")
            self.assertEqual(list((project.out / "runs").glob("*")), [])
        finally:
            engine.LOCK = real

    def test_the_arms_error_is_named_with_the_scripts_own_words(self):
        # scripts/preflight.py says which joint detection gave up on, and how big the model is
        said = json.dumps({"code": "VM-ARMS-AT-SIDES", "detail": "the arms don't stand clear of the body (the left one); and 1,900,000 triangles is heavy"})
        known = errors.classify("survey", "MODEL pirate.fbx\nVTUBER_MAKER_ERROR " + said + "\n")
        self.assertEqual((known["code"], known["options"]), ("VM-ARMS-AT-SIDES", None))
        self.assertIn("T-pose", known["fix"])
        self.assertIn("1,900,000 triangles", known["detail"])
        # ... and the build, which has no such line, is known by the reason alone
        self.assertEqual(errors.classify("build", "LandmarkError: the arms don't stand clear of the body: they hang")["code"], "VM-ARMS-AT-SIDES")
        self.assertEqual(errors.classify("survey", "This doesn't read as a standing humanoid: could not find two separate legs")["code"], "VM-NOT-HUMANOID")

    def test_a_model_the_app_turned_around_is_built_as_its_file_has_it(self):
        project = self.project()
        self.assertEqual(project.frame(), [])
        project = self.project(facing="file")
        self.assertEqual(project.frame(), ["--python-expr", "import os; os.environ['VTUBER_MAKER_FACING'] = 'file'"])
        data = json.loads(self.path.read_text())
        data["options"]["facing"] = "back"
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "option facing must be one of"):
            manifest.load(self.path)

    def vrm_0(self, licence="Other", page="https://example.org/terms.html", permission=None):
        """A small VRM 0.x file's JSON: three bones, a mouth shape and a blink, two springs, a toon material."""
        meta = {"title": "Old", "author": "Somebody Else", "licenseName": licence, "otherLicenseUrl": page,
                **({"otherPermissionUrl": permission} if permission else {})}
        groups = [{"name": "A", "presetName": "a", "binds": [{"mesh": 0, "index": 0, "weight": 100}]},
                  {"name": "Blink", "presetName": "blink", "binds": [{"mesh": 0, "index": 1, "weight": 100}]},
                  {"name": "LookUp", "presetName": "lookup", "binds": []}]
        return {"asset": {"version": "2.0"}, "nodes": [{"name": "Hips", "children": [1]}, {"name": "Head", "children": [2]}, {"name": "Index"}],
                "skins": [{"joints": [0, 1, 2]}], "images": [{}, {}], "materials": [{}, {}],
                "meshes": [{"primitives": [{"extras": {"targetNames": ["Fcl_A", "Fcl_Blink"]}}]}],
                "extensions": {"VRM": {"specVersion": "0.0", "meta": meta,
                                       "humanoid": {"humanBones": [{"bone": "hips", "node": 0}, {"bone": "head", "node": 1}, {"bone": "leftIndexProximal", "node": 2}]},
                                       "blendShapeMaster": {"blendShapeGroups": groups}, "firstPerson": {"lookAtTypeName": "Bone"},
                                       "secondaryAnimation": {"boneGroups": [{"bones": [1, 2]}, {"bones": []}],
                                                              "colliderGroups": [{"node": 1, "colliders": [{}, {}]}, {"node": 0, "colliders": [{}]}]},
                                       "materialProperties": [{"shader": "VRM/MToon"}, {"shader": "VRM_USE_GLTFSHADER"}]}}}

    def test_a_vrm_0_file_is_read_in_the_same_words_as_a_vrm_1(self):
        holds = engine.shipped(Engine.write_glb(self, "old.vrm", self.vrm_0()))
        self.assertEqual({k: holds[k] for k in ("bones", "fingers", "blink", "vowels", "look", "looks_bound", "shapes", "springs", "colliders",
                                                "materials", "toon", "textures", "turned_joints", "arms_off_level")},
                         {"bones": 3, "fingers": 1, "blink": True, "vowels": ["aa"], "look": "bone", "looks_bound": False,
                          "shapes": ["Fcl_A", "Fcl_Blink"], "springs": 2, "colliders": 3, "materials": 2, "toon": 1, "textures": 2,
                          "turned_joints": [], "arms_off_level": None})
        # held against the 1.0 file it leaves as: nothing of its own counts as lost for being 0.x
        self.assertEqual(engine.lost(holds, {**holds, "megabytes": 1}), [])
        self.assertIn("1 springs, the file taken in had 2", engine.lost(holds, {**holds, "springs": 1})[0])
        self.assertIsNone(engine.shipped(Engine.write_glb(self, "no.vrm", {"asset": {}, "extensions": {"VRM": {"specVersion": "0.0"}}})))

    def test_whose_avatar_it_is_and_whether_its_terms_have_to_be_read(self):
        licence = lambda **said: engine.licence(Engine.write_glb(self, "old.vrm", self.vrm_0(**said)))  # noqa: E731
        other = licence()
        self.assertEqual((other["spec"], other["name"], other["author"], other["pages"]), ("0.x", "Old", "Somebody Else", ["https://example.org/terms.html"]))
        self.assertTrue(other["confirm"])
        self.assertIn("https://example.org/terms.html", other["why"])
        # a licence that says changed versions are fine asks nothing, one that forbids them asks
        self.assertFalse(licence(licence="CC0")["confirm"])
        self.assertFalse(licence(licence="CC_BY")["confirm"])
        self.assertIn("doesn't allow changed versions", licence(licence="CC_BY_ND")["why"])
        self.assertIn("no address", licence(page=None)["why"])
        # VRoid Hub's licence page says it in its address: read, unless it says no changes
        hub = "https://hub.vroid.com/en/license?modification={}&version=1"
        self.assertFalse(licence(page=hub.format("allow"))["confirm"])
        self.assertTrue(licence(page=hub.format("disallow"))["confirm"])
        self.assertTrue(licence(licence="CC0", permission="https://example.org/more.html")["confirm"])
        # a VRM 1.0 file's terms are not judged, as the add-on doesn't; a plain model has none
        own = engine.licence(Engine.write_glb(self, "new.vrm", Engine.avatar(self)))
        self.assertEqual((own["spec"], own["confirm"]), ("1.0", False))
        self.assertIsNone(engine.licence(Engine.write_glb(self, "plain.glb", {"asset": {}})))

    def test_a_vrm_0_file_makes_a_character_and_its_licence_is_asked_for(self):
        old = Engine.write_glb(self, "old.vrm", self.vrm_0())
        made = cli.cmd_init(SimpleNamespace(model=old, name="Old", author="Unknown", face=None, reference=None, rig=None,
                                            manifest=self.root / "old.json"))
        self.assertTrue(made["improving"] and made["licence"]["confirm"])
        project = Project(self.root / "old.json", blender="blender")
        build = lambda p: p._arguments("build", self.root / "runs" / "01_build", None)[0]  # noqa: E731
        self.assertNotIn("--licence-read", build(project))
        cli.cmd_set(SimpleNamespace(manifest=self.root / "old.json", key="licence_read", value="true"))
        self.assertIn("--licence-read", build(Project(self.root / "old.json", blender="blender")))
        known = errors.classify("build", 'VTUBER_MAKER_ERROR {"code": "VM-LICENCE", "detail": "terms (https://example.org/terms.html)"}')
        self.assertEqual((known["code"], known["detail"]), ("VM-LICENCE", "terms (https://example.org/terms.html)"))
        self.assertIn("Options page", known["fix"])

    def test_a_step_that_has_not_run_is_said_not_a_key_error(self):
        project = self.project()
        for step in ("tpose", "survey"):
            with self.assertRaisesRegex(engine.StepFailed, f"{step} has no .blend yet"):
                cli._blend_of(project, step)
        project.state["steps"]["tpose"] = {"status": "done", "blend": str(self.root / "13_tpose.blend")}
        self.assertEqual(cli._blend_of(project, "tpose"), self.root / "13_tpose.blend")

    def test_an_installers_output_is_read_whatever_its_bytes(self):
        # UTF-8 from pip, read in Windows' code page: 0x81 and 0x8d have no letter in cp1252
        printed = "import sys; sys.stdout.buffer.write(b'C:/Users/J\\x81\\x8d\\xc3\\xa9/site-packages ok')"
        done = engine.install([sys.executable, "-c", printed], "a package")
        self.assertIn("site-packages ok", done["log"])

    def test_a_lock_left_by_a_dead_process_is_taken(self):
        import subprocess
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        real, engine.LOCK = engine.LOCK, self.root / ".blender.lock"
        try:
            engine.LOCK.write_text(f"pid {dead.pid} since yesterday")
            with engine._Lock():
                self.assertIn(f"pid {engine.os.getpid()}", engine.LOCK.read_text())
            engine.LOCK.write_text(f"pid {engine.os.getpid()} since now")  # alive: still refused
            with self.assertRaisesRegex(engine.StepFailed, "another build is running"):
                with engine._Lock():
                    pass
        finally:
            engine.LOCK.unlink(missing_ok=True)
            engine.LOCK = real

    def test_known_failures_name_their_code_and_fix(self):
        self.assertEqual(errors.classify("tpose", "Writing\nMalloc returns null\n")["code"], "VM-OOM")
        eyes = errors.classify("eyes", "wrote x\nFAILED: left iris over the lids\n")
        self.assertEqual((eyes["code"], eyes["options"]), ("VM-EYES-REFUSED", {"eyes": "slide"}))
        self.assertIsNone(errors.classify("tpose", "FAILED: something"))  # the eyes' refusal only on the eye step
        named = errors.classify("build", 'VTUBER_MAKER_ERROR {"code": "VM-OOM", "detail": "proxy 512"}')
        self.assertEqual((named["code"], named["detail"]), ("VM-OOM", "proxy 512"))
        self.assertIsNone(errors.classify("build", "Traceback: KeyError"))
        # the exported-file check's own command line names the import operator: not a missing add-on
        echoed = 'blender -b --python-expr "assert bpy.ops.import_scene.vrm(filepath=\'a.vrm\')" -P check_avatar.py\nKeyError: 1\n'
        self.assertIsNone(errors.classify("vrm_check", echoed))
        for said in ('AttributeError: Calling operator "bpy.ops.export_scene.vrm" error, could not be found',
                     "vtuber_maker.pipeline.vrm_export.VrmAddonMissing: Install 'VRM format'"):
            self.assertEqual(errors.classify("tpose", said)["code"], "VM-NO-VRM-ADDON")
        moved = "vtuber_maker.pipeline.vrm_export.ExportMismatch: a.vrm: 30 joints aren't where the rest pose has them"
        self.assertEqual(errors.classify("tpose", moved)["code"], "VM-EXPORT-POSE")

    def test_a_failure_that_stops_the_chain_rides_beside_the_last_good_output(self):
        project = self.project({"options": {}, "triangles": 40000})
        project._record("build", {"status": "done", "blend": "good.blend"})

        def fails(name, index, folder):
            raise engine.StepFailed("build: failed", errors.describe("VM-OOM"))
        project._run_step = fails
        real = engine.LOCK
        engine.LOCK = self.root / ".blender.lock"
        try:
            with self.assertRaises(engine.StepFailed):
                project.run(only="build")
        finally:
            engine.LOCK = real
        build = project.state["steps"]["build"]
        self.assertEqual((build["status"], build["blend"]), ("done", "good.blend"))
        self.assertEqual(build["last_failure"]["known"]["code"], "VM-OOM")
        shown = next(s for s in Project(self.path).status()["steps"] if s["step"] == "build")
        self.assertIn("last_failure", shown)
        # every run says what built it
        made_by = json.loads(next((project.out / "runs").glob("*/run.json")).read_text())
        self.assertEqual(made_by["steps"], ["build"])
        self.assertIn("commit", made_by)
        project._record("build", {"status": "done", "blend": "new.blend"})
        self.assertNotIn("last_failure", project.state["steps"]["build"])

    def fingerprint(self, **changed):
        """A build's numbers after the face step, as scripts/_fingerprint.py writes them."""
        found = {"meshes": {"Body": {"vertices": 27786, "polygons": 52418, "materials": {"skin": 52338, "mouth": 80}}},
                 "keys": {"Body": {"aa": {"moved": 99, "max_mm": 15.98}, "eyeLookInLeft": {"moved": 289, "max_mm": 2.91}}},
                 "face": {"Body": {"vertices": 1439, "head": 1.0, "on_springs": 0}},
                 "fingers": {"LeftThumbProximal": 22.9, "LeftIndexProximal": 2.9}, "bones": 105, "spring_bones": 55}
        for path, value in changed.items():
            *parents, last = path.split("__")
            at = found
            for name in parents:
                at = at[name]
            at[last] = value
        return found

    def test_a_step_may_only_change_what_is_its_own(self):
        compare, good = engine.contracts.compare, self.fingerprint()
        self.assertEqual(compare("tpose", good, self.fingerprint()), ([], []))
        self.assertEqual(compare("loose", None, good), ([], []))
        self.assertEqual(compare("loose", good, None), ([], []))  # a step whose numbers weren't written
        # the loose step welded the mouth patch: half the vertices is what joining twins does, the reach isn't
        joined = self.fingerprint(keys__Body__aa={"moved": 60, "max_mm": 15.98}, meshes__Body__vertices=20000)
        self.assertEqual(compare("loose", good, joined)[0], [])
        welded = self.fingerprint(keys__Body__aa={"moved": 36, "max_mm": 11.37})
        self.assertIn("aa reaches 11.37 mm, was 15.98", compare("loose", good, welded)[0][0])
        self.assertEqual(compare("mouth", good, welded)[0], [])  # the mouth step reshapes the mouth
        # a hand import emptied the look keys: no step may, the eye step included
        wiped = self.fingerprint(keys__Body__eyeLookInLeft={"moved": 0, "max_mm": 0.0})
        for step in ("pieces", "eyes"):
            self.assertIn("eyeLookInLeft no longer moves anything", compare(step, good, wiped)[0][0])
        # the T-pose step relays the fingers of our own rig, and leaves an adopted rig's as modelled
        fanned = self.fingerprint(fingers__LeftThumbProximal=45.0)
        self.assertEqual(compare("tpose", good, fanned)[0], [])
        self.assertIn("LeftThumbProximal most: 22.9 -> 45.0", compare("tpose", good, fanned, rigged=True)[0][0])
        self.assertTrue(compare("chest", good, fanned)[0])
        # nothing but the steps that cut and join touches the mesh
        cut = self.fingerprint(meshes__Body__polygons=52372)
        self.assertTrue(compare("tpose", good, cut)[0])
        self.assertEqual(compare("eyes", good, cut)[0], [])
        self.assertEqual(compare("chest", good, self.fingerprint(spring_bones=59))[0], [])
        self.assertIn("51 spring bones, was 55", compare("eyes", good, self.fingerprint(spring_bones=51))[0])
        # the face step gives skin it eased back over the mouth the face's picture (the miko's lip), nothing else does
        repainted = self.fingerprint(meshes__Body__materials={"skin": 52348, "mouth": 70})
        self.assertEqual(compare("face_tracking", good, repainted)[0], [])
        self.assertTrue(compare("chest", good, repainted)[0])
        self.assertTrue(compare("face_tracking", good, cut)[0])

    def test_a_step_that_cuts_nothing_may_not_open_the_surface(self):
        # one copy of a seam's vertices moved and not its twin: 92 border edges more across model A's lower back
        compare = engine.contracts.compare
        whole, slit = self.fingerprint(meshes__Body__open_edges=3132), self.fingerprint(meshes__Body__open_edges=3224)
        self.assertIn("92 edges that were joined at rest are open now", compare("chest", whole, slit)[0][0])
        # a few places either side of the count's rounding are none, and a step that cuts may open what it likes
        self.assertEqual(compare("chest", whole, self.fingerprint(meshes__Body__open_edges=3138))[0], [])
        self.assertEqual(compare("eyes", whole, slit)[0], [])
        # the bake parts the two sides of every sleeve cut (47 edges on model A built without her skirt's repair,
        # which this count failed as a slit): the T-pose step counts its own, copy by copy (loose.parted_copies)
        self.assertEqual(compare("tpose", whole, self.fingerprint(meshes__Body__open_edges=3179))[0], [])
        self.assertEqual(compare("tpose", whole, slit)[0], [])
        # a build from before the count was kept
        self.assertEqual(compare("chest", self.fingerprint(), slit)[0], [])

    def test_pieces_get_roles_by_themselves_only_when_asked(self):
        survey = {"options": {}, "triangles": 40000}
        self.assertFalse(manifest.OPTIONS["auto_pieces"])
        self.assertNotIn("pieces", self.runs(self.project(survey)))
        project = self.project(survey, auto_pieces=True)
        self.assertIn("pieces", self.runs(project))
        # the step reads the user's manifest until the movement check has found pieces of its own
        stem = self.root / "07_pieces"
        self.assertEqual(project._arguments("pieces", stem, self.root / "in.blend")[0][-1], project.manifest_path)
        engine._found_manifest(stem).write_text("{}")
        self.assertEqual(project._arguments("pieces", stem, self.root / "in.blend")[0][-1], engine._found_manifest(stem))
        # ... and the movement check after the build names pieces by the roles that step gave
        named = lambda: project._arguments("movement", self.root / "16_movement", self.root / "in.blend")[0]  # noqa: E731
        self.assertEqual(named()[named().index("--manifest") + 1], project.manifest_path)
        project.state["steps"]["pieces"] = {"status": "done", "report": {"manifest": str(engine._found_manifest(stem)), "found": ["follow_bone_7"]}}
        self.assertEqual(named()[named().index("--manifest") + 1], engine._found_manifest(stem))
        project.state["steps"]["pieces"]["report"]["found"] = []
        self.assertEqual(named()[named().index("--manifest") + 1], project.manifest_path)
        data = manifest.load(self.path)
        data["options"]["auto_pieces"] = "yes"
        with self.assertRaisesRegex(ValueError, "auto_pieces"):
            manifest.validate(data)

    def test_what_the_movement_check_found_becomes_pieces(self):
        # model B without the pieces named by hand: a hanging sleeve and the mantle, as "Apply this fix" writes them
        behind = lambda bone: [{"pose": "arms raised", "kind": "stays behind", "bone": bone, "taken_by": 3, "share": 0.7}]  # noqa: E731
        movement = {"pieces": [
            {"label": 9041, "count": 158, "point": [0.162, -0.046, 0.997], "suggested": "follow_bone", "findings": behind("LeftUpperArm")},
            {"label": 5120, "count": 1039, "point": [-0.24, -0.107, 1.106], "suggested": "shoulder_cloth", "findings": behind(None)},
            # a role the build step acts on is the user's to give, a piece the user named keeps its role, and
            # a finding with no role for it (the waist piece a hanging hand rested on) is left alone
            {"label": 77, "count": 400, "point": [0.0, 0.1, 1.5], "suggested": "hair", "findings": behind("Head")},
            {"label": 12, "count": 13569, "point": [-0.17, -0.086, 0.804], "suggested": "fitted_skirt", "role": "body",
             "name": "skirt", "findings": [{"pose": "arms raised", "kind": "torn", "stretch_cm": 41.3}]},
            {"label": 5, "count": 762, "point": [0.1, -0.1, 0.9], "suggested": "", "findings": behind("RightHand")}]}
        found = engine.found_pieces(movement, [{"name": "skirt"}, {"name": "shoulder_cloth_5120"}])
        self.assertEqual([(p["name"], p["role"], p.get("bone"), p["count"], p["found"]) for p in found],
                         [("follow_bone_9041", "follow_bone", "LeftUpperArm", 158, True),
                          ("shoulder_cloth_5120_found", "shoulder_cloth", None, 1039, True)])
        data = manifest.load(self.path)
        data["pieces"] = found
        manifest.validate(data)
        self.assertEqual(engine.found_pieces({}, []), [])

    def test_face_pieces_on_hair_springs_are_said_once(self):
        compare = engine.contracts.compare
        swinging = self.fingerprint(face__Body={"vertices": 1970, "head": 0.72, "on_springs": 328})
        damage, warnings = compare("face_tracking", self.fingerprint(face__Body={"vertices": 0, "head": None, "on_springs": 0}), swinging)
        self.assertEqual(damage, [])
        self.assertEqual(len(warnings), 2)
        self.assertIn("328 of the 1970", warnings[0])
        # the loose step joins vertices and the counts change: the same finding, not a new one
        later = self.fingerprint(face__Body={"vertices": 1376, "head": 0.71, "on_springs": 232})
        self.assertEqual(compare("loose", swinging, later), ([], []))

    def test_a_step_that_damages_another_steps_work_fails(self):
        project = self.project({"options": {}, "triangles": 40000})
        before = self.root / "08_before.blend"
        before.write_bytes(b"BLENDER-v402")
        engine._fingerprint_of(before).write_text(json.dumps(self.fingerprint()))
        project._record("loose", {"status": "done", "blend": str(before)})
        project.data["rig"] = "rigged.fbx"
        real, seen = engine._blender, {}

        def blender(blender, blend_in, args, log, extra):
            seen["extra"] = [str(e) for e in extra]
            Path(log).write_text("")
            Path(args[1]).write_bytes(b"BLENDER-v402")
            engine._fingerprint_of(args[1]).write_text(json.dumps(self.fingerprint(fingers__LeftThumbProximal=seen["thumb"])))
            return 0
        engine._blender = blender
        try:
            seen["thumb"] = 22.9
            entry = project._run_step("tpose", engine.NAMES.index("tpose"), self.root)
            self.assertEqual(entry["status"], "done")
            self.assertEqual(seen["extra"][-1], str(engine.FINGERPRINT))
            self.assertIn("09_tpose_fingerprint.json", seen["extra"][-3])
            seen["thumb"] = 45.0
            with self.assertRaises(engine.StepFailed) as failed:
                project._run_step("tpose", engine.NAMES.index("tpose"), self.root)
            self.assertEqual(failed.exception.known["code"], "VM-STEP-DAMAGE")
            self.assertIn("22.9 -> 45.0", str(failed.exception))
        finally:
            engine._blender = real

    def _failing_blender(self, fail, seen):
        """A stand-in for a Blender: the step's script fails while `fail(arguments)` says so, else writes its file."""
        def blender(blender, blend_in, args, log, extra):
            seen.append([str(a) for a in args])
            if fail(seen[-1]):
                Path(log).write_text("Traceback (most recent call last):\nValueError: need at least two points\n")
                return 1
            Path(log).write_text("")
            Path(args[args.index("--blend") + 1] if "--blend" in args else args[1]).write_bytes(b"BLENDER-v402")
            return 0
        return blender

    def test_a_build_that_fails_on_the_model_is_made_the_plain_way(self):
        # the user, 2026-11-10: nearly every new model broke the build in a way of its own, and ended with no file
        project = self.project({"options": {}, "triangles": 40000})
        real, seen, said = engine._blender, [], []
        engine._blender = self._failing_blender(lambda args: "--plain" not in args, seen)
        try:
            entry = project._step_or_plainer("build", engine.NAMES.index("build"), self.root, said.append)
        finally:
            engine._blender = real
        self.assertEqual(entry["status"], "done")
        self.assertEqual(["--plain" in args for args in seen], [False, True])
        self.assertEqual(entry["report"]["plainer"], "plain")
        self.assertTrue(entry["report"]["asked_failed"])
        self.assertIn("plain way", entry["warnings"][-1])
        self.assertEqual([event["status"] for event in said], ["warning"])
        # the log of the build as asked is kept beside the plain one's
        self.assertIn("ValueError", (self.root / "01_build_asked.log").read_text())
        # ... and the verdict says how it was built
        project._record("build", entry)
        project._record("vrm_check", {"status": "done"})
        self.assertTrue(any("plain way" in line for _, line in project.verdict()["looks"]))

    def test_the_plain_build_tries_the_other_skinning_last_and_gives_the_first_failure_up(self):
        project = self.project({"options": {}, "triangles": 40000})
        real, seen = engine._blender, []
        engine._blender = self._failing_blender(lambda args: args[args.index("--skinning") + 1] != "voxel", seen)
        try:
            entry = project._step_or_plainer("build", engine.NAMES.index("build"), self.root, lambda event: None)
            self.assertEqual(entry["report"]["plainer"], "plain_voxel")
            self.assertEqual(len(seen), 3)
            del seen[:]
            engine._blender = self._failing_blender(lambda args: True, seen)
            with self.assertRaises(engine.StepFailed) as failed:
                project._step_or_plainer("build", engine.NAMES.index("build"), self.root, lambda event: None)
            self.assertEqual(len(seen), 3)
            self.assertNotIn("--plain", " ".join(str(failed.exception).split()[:3]))
            # the next run starts as asked again
            self.assertNotIn("--plain", [str(a) for a in project._arguments("build", self.root / "01_build", None)[0]])
        finally:
            engine._blender = real

    def test_a_failure_that_is_not_the_models_is_not_tried_a_plainer_way(self):
        project = self.project({"options": {}, "triangles": 40000})
        real, seen = engine._blender, []

        def blender(blender, blend_in, args, log, extra):
            seen.append(args)
            Path(log).write_text("Malloc returns null\n")
            return 1
        engine._blender = blender
        try:
            with self.assertRaises(engine.StepFailed) as failed:
                project._step_or_plainer("build", engine.NAMES.index("build"), self.root, lambda event: None)
        finally:
            engine._blender = real
        self.assertEqual(failed.exception.known["code"], "VM-OOM")
        self.assertEqual(len(seen), 1)

    def test_no_step_after_the_build_ends_it_but_the_bake_and_the_export(self):
        fatal = [name for name in engine.NAMES if name not in engine.CARRIES_ON | engine.OPTIONAL]
        self.assertEqual(fatal, ["survey", "build", "tpose", "export", "vrm_check"])
        self.assertEqual(set(engine.PLAINER), {"survey", "build", "tpose"})
        # a rig check that fell over leaves the file, and holds it back
        project = self.project({"options": {}, "triangles": 40000})
        project._record("check", {"status": "failed", "reason": "check: failed"})
        project._record("vrm_check", {"status": "done"})
        verdict = project.verdict()
        self.assertFalse(verdict["ready"])
        self.assertTrue(any("rig check couldn't run" in line for _, line in verdict["issues"]))

    def test_contact_sheets_come_back_from_the_browser(self):
        import subprocess
        import threading
        import urllib.request
        from app import sheet
        vrm = self.root / "avatar.vrm"
        vrm.write_bytes(b"glTF-model")
        seen = {}

        class Browser(subprocess.Popen):
            """Stands in for headless Edge: loads the page and the model, sends two sheets back. A process that
            waits, so sheet.close() ends a real one (and the tree it would have started) as it does the browser."""
            def __init__(self, command, **kwargs):
                self.stands_in = command[0] == "edge.exe"
                if not self.stands_in:
                    # what sheet.close() itself runs to end the browser: as it is. Stood in for too,
                    # taskkill "was" a process sleeping two minutes
                    return super().__init__(command, **kwargs)
                seen["command"], seen["process"] = command, self
                super().__init__([sys.executable, "-c", "import time; time.sleep(120)"], **kwargs)
                threading.Thread(target=self.visit, args=(command[-1],), daemon=True).start()

            def visit(self, url):
                root = url.split("/viewer/")[0]
                seen["page"] = urllib.request.urlopen(url).read()
                seen["model"] = urllib.request.urlopen(root + "/model.vrm").read()
                for name, body in (("face", b"PNG-face"), ("../escape", b"no"), (self.last, json.dumps(self.said).encode())):
                    urllib.request.urlopen(urllib.request.Request(f"{root}/sheet?name={name}", data=body, method="POST"))

            def kill(self):
                seen["killed"] = seen.get("killed") or self.stands_in
                super().kill()
        real = sheet.subprocess.Popen, sheet.find_browser
        sheet.subprocess.Popen, sheet.find_browser = Browser, lambda given=None: "edge.exe"
        try:
            Browser.last, Browser.said = "done", {"missing": ["blinkLeft"], "warnings": []}
            result = sheet.render(vrm, self.root / "sheets")
            self.assertEqual([Path(p).name for p in result["images"]], ["sheet_face.png"])
            self.assertEqual((self.root / "sheets" / "sheet_face.png").read_bytes(), b"PNG-face")
            self.assertEqual((seen["model"], result["report"]["missing"]), (b"glTF-model", ["blinkLeft"]))
            self.assertIn(b"three-vrm", seen["page"])
            # a profile of its own, by its whole path (Edge shows a dialog for a relative one), gone afterwards
            profile = Path(next(c for c in seen["command"] if c.startswith("--user-data-dir=")).split("=", 1)[1])
            self.assertTrue(profile.is_absolute() and not profile.exists() and seen["killed"])
            self.assertIsNotNone(seen["process"].returncode)
            # a texture the browser couldn't load is the page's failure, not a sheet of a white body
            Browser.last, Browser.said = "failed", {"error": "a texture didn't load"}
            with self.assertRaises(RuntimeError):
                sheet.render(vrm, self.root / "sheets")
        finally:
            sheet.subprocess.Popen, sheet.find_browser = real

    def test_a_browser_that_never_comes_for_its_page_is_started_again(self):
        # one start of the browser held a build for its whole five minutes: nothing asked for, nothing drawn
        import subprocess
        import threading
        import urllib.request
        from app import sheet
        vrm = self.root / "avatar.vrm"
        vrm.write_bytes(b"glTF-model")
        started = []

        class Browser(subprocess.Popen):
            comes = True

            def __init__(self, command, **kwargs):
                if command[0] != "edge.exe":
                    return super().__init__(command, **kwargs)  # what sheet.close() runs to end a browser
                started.append(self)
                super().__init__([sys.executable, "-c", "import time; time.sleep(120)"], **kwargs)
                if self.comes and len(started) > 1:  # the first one sits there
                    threading.Thread(target=self.visit, args=(command[-1],), daemon=True).start()

            def visit(self, url):
                urllib.request.urlopen(url).read()
                said = urllib.request.Request(url.split("/viewer/")[0] + "/sheet?name=done", data=b'{"missing": []}', method="POST")
                urllib.request.urlopen(said)
        real = sheet.subprocess.Popen, sheet.find_browser, sheet.NEVER_ASKED
        sheet.subprocess.Popen, sheet.find_browser, sheet.NEVER_ASKED = Browser, lambda given=None: "edge.exe", 0.3
        try:
            self.assertEqual(sheet.render(vrm, self.root / "sheets")["report"]["missing"], [])
            self.assertEqual(len(started), 2)
            self.assertTrue(all(browser.returncode is not None for browser in started))  # both ended, the first too
            # one that never comes is given up on when the time is up, and what was seen is said
            Browser.comes = False
            del started[:]
            with self.assertRaisesRegex(TimeoutError, "the browser never asked for the page, started 3 times"):
                sheet.browse(sheet.PAGE, vrm, self.root / "sheets", timeout=3)
            self.assertEqual(len(started), sheet.STARTS)
            self.assertEqual(list((self.root / "sheets").glob(".browser_*")), [])
        finally:
            sheet.subprocess.Popen, sheet.find_browser, sheet.NEVER_ASKED = real

    def test_the_browsers_profile_lies_where_the_browser_can_write_in_it(self):
        # Edge writes files 120 characters below its profile and none past Windows' 260: with its profile
        # in a deep run folder it never came for its page, and every build waited five minutes for it
        from unittest import mock
        from app import sheet
        beside = Path(sheet.profile_for(self.root / "sheets"))
        self.assertEqual((beside.parent, beside.is_dir(), beside.name[:9]), (self.root / "sheets", True, ".browser_"))
        if os.name != "nt":
            return  # Windows' own limit
        room = lambda place: sheet.WINDOWS_PATH - sheet.PROFILE_ROOM - sheet.PROFILE_NAME - len(str(place))  # noqa: E731
        deep = self.root / ("d" * room(self.root))  # with its separator, one character too deep
        self.assertEqual(room(deep), -1)
        with mock.patch.object(sheet, "REPO", self.root / "app"):
            # ... the app's own out folder then, made if it isn't there
            self.assertEqual(Path(sheet.profile_for(deep)).parent, self.root / "app" / "out")
            self.assertFalse(deep.exists())
            # a folder with just the room is still the place
            fits = self.root / ("d" * (room(self.root) - 1))
            self.assertEqual((room(fits), Path(sheet.profile_for(fits)).parent), (0, fits))
        temp = self.root / "temp"
        temp.mkdir()
        with mock.patch.object(sheet, "REPO", deep), mock.patch.object(sheet.tempfile, "gettempdir", return_value=str(temp)):
            # the app itself lies deep: the temp folder
            self.assertEqual(Path(sheet.profile_for(deep / "runs" / "x")).parent, temp)
        with mock.patch.object(sheet, "REPO", deep / "app"), mock.patch.object(sheet.tempfile, "gettempdir", return_value=str(deep / "t")):
            # nowhere with room: the shallowest of them, since a browser with little room still draws
            self.assertEqual(Path(sheet.profile_for(deep / "runs" / "x")).parent, deep / "t")

    def test_the_browsers_profile_goes_where_the_drive_has_room_for_its_textures(self):
        # on an E: with 4 GB free the browser refused a texture and drew nothing, while C: had 9: the fox girl
        # passed with her moving iris over a hole, KemonoGirl unlooked at (2026-10-16)
        from unittest import mock
        from app import sheet
        out, temp = self.root / "run" / "x_seen", self.root / "temp"
        temp.mkdir()
        free = {out: 4 << 30, self.root / "app" / "out": 4 << 30, temp: 9 << 30}
        with mock.patch.object(sheet, "REPO", self.root / "app"), mock.patch.object(sheet.tempfile, "gettempdir", return_value=str(temp)), \
                mock.patch.object(sheet, "free_at", side_effect=lambda place: free[Path(place)]):
            self.assertEqual(sheet.profile_places(out)[0], temp)
            self.assertEqual(Path(sheet.profile_for(out)).parent, temp)
            # room beside the sheets: there, as before
            free[out] = 6 << 30
            self.assertEqual(sheet.profile_places(out), [out, temp, self.root / "app" / "out"])
            # room nowhere: the order as it was
            free[out], free[temp] = 4 << 30, 4 << 30
            self.assertEqual(sheet.profile_places(out), [out, self.root / "app" / "out", temp])
        self.assertGreater(sheet.free_at(self.root / "not" / "made" / "yet"), 0)  # read on the drive it would be on

    def landmarks(self, mouth_height=0.0):
        """What the detector found on model A (T-pose)'s face patch picture, the mouth's height set."""
        at = {11: (697, 835), 12: (791, 804), 13: (901, 853), 15: (810, 932), 17: (1112, 853), 18: (1221, 800),
              19: (1317, 838), 21: (1216, 936), 24: (958, 1107), 25: (1006, 1104 - mouth_height / 2),
              26: (1050, 1107), 27: (1006, 1104 + mouth_height / 2)}
        return {"size": [2048, 2048], "faces": [{"box": [613, 559, 1395, 1325, 1.0],
                                                  "points": [[*at.get(i, (0, 0)), 0.93] for i in range(28)]}]}

    def test_a_detected_face_becomes_the_twelve_points(self):
        from app import face_guess
        points = face_guess.from_landmarks(self.landmarks())
        # image left is the character's right eye; corners, top and the middle of the lower lid
        self.assertEqual((points["right_eye"]["outer"], points["right_eye"]["inner"]), ([697, 835], [901, 853]))
        self.assertEqual((points["left_eye"]["inner"], points["left_eye"]["bottom"]), ([1112, 853], [1216, 936]))
        # a mouth drawn as a line: the corners stay, top and bottom go above and below its middle
        mouth = points["mouth"]
        self.assertEqual((mouth["left"], mouth["right"]), ([958, 1107], [1050, 1107]))
        self.assertAlmostEqual(mouth["top"][1], 1104 - 0.25 * 92)
        self.assertAlmostEqual(mouth["bottom"][1], 1104 + 0.3 * 92)
        self.assertEqual((mouth["top"][0], points["sure"]), (1004.0, 0.93))
        # an open mouth keeps the detector's own top and bottom
        self.assertEqual(face_guess.from_landmarks(self.landmarks(mouth_height=40))["mouth"]["top"], [1006, 1084.0])
        # every point the app's face.json needs is there, so it saves as hand-placed ones do
        for key, names in cli.FACE_KEYS.items():
            self.assertEqual(set(points[key]), set(names))
        unsure = self.landmarks()
        unsure["faces"][0]["box"][4] = 0.2
        # a face found, its points not (model B): no guess is better than one two eye widths off
        vague = self.landmarks()
        vague["faces"][0]["points"][18][2] = 0.1
        for found in (unsure, vague, {"size": [2048, 2048], "faces": []}):
            with self.assertRaises(face_guess.NoFace):
                face_guess.from_landmarks(found)

    def test_the_guess_runs_in_a_python_that_has_the_detector(self):
        from app import face_guess
        ran = {}

        def run(command, **keywords):
            ran["command"], ran["env"] = command, keywords.get("env")
            return SimpleNamespace(returncode=0, stderr="", stdout="loading" + chr(10) + "FACE_LANDMARKS " + json.dumps(self.landmarks()) + chr(10))
        real = face_guess.subprocess.run, face_guess.find_python
        face_guess.subprocess.run, face_guess.find_python = run, lambda given=None: given or "ai-python"
        try:
            points = face_guess.guess(self.root / "head.png", cache=self.root / "hf")
            self.assertEqual(points["left_eye"]["top"], [1221, 800])
            self.assertEqual(ran["command"], ["ai-python", str(face_guess.SCRIPT), str(self.root / "head.png")])
            self.assertEqual(ran["env"]["HF_HOME"], str(self.root / "hf"))
            face_guess.subprocess.run = lambda command, **keywords: SimpleNamespace(returncode=1, stderr="No module named torch", stdout="")
            with self.assertRaises(RuntimeError):
                face_guess.guess(self.root / "head.png")
        finally:
            face_guess.subprocess.run, face_guess.find_python = real
        # without a face patch there is no picture to look at
        with self.assertRaises(engine.StepFailed):
            cli.guess_face(self.path)

    def test_outputs_are_checked_where_they_are_made(self):
        def glb(bones):
            gltf = json.dumps({"nodes": [{}] * 20, "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": bones}}}}).encode()
            gltf += b" " * (-len(gltf) % 4)
            path = self.root / "a.vrm"
            path.write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(gltf)) + struct.pack("<I", len(gltf)) + b"JSON" + gltf)
            return str(path)
        blend = self.root / "a.blend"
        blend.write_bytes(b"BLENDER-v402")
        every = {b: {"node": i} for i, b in enumerate(sorted(engine.REQUIRED_BONES))}
        self.assertIsNone(engine.output_problem({"blend": str(blend), "vrm": glb(every)}))
        self.assertIn("leftFoot", engine.output_problem({"vrm": glb({k: v for k, v in every.items() if k != "leftFoot"})}))
        self.assertIn("no node", engine.output_problem({"vrm": glb({**every, "head": {"node": 99}})}))
        self.assertIn("wasn't exported", engine.output_problem({"vrm": str(self.root / "missing.vrm")}))
        blend.write_bytes(b"")  # truncated
        self.assertIn("isn't a .blend", engine.output_problem({"blend": str(blend)}))

    def test_a_skeleton_the_mesh_isnt_bound_to_or_that_cant_exist_is_refused(self):
        names = sorted(engine.REQUIRED_BONES)

        def vrm(joints, binds):
            gltf = {"nodes": [{"name": n} for n in names], "skins": [{"joints": joints, "inverseBindMatrices": 0}],
                    "accessors": [{"bufferView": 0, "count": len(joints), "type": "MAT4"}], "bufferViews": [{"byteOffset": 0}],
                    "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": {n: {"node": i} for i, n in enumerate(names)}}}}}
            return {"vrm": str(self.write_glb("a.vrm", gltf, struct.pack(f"<{len(binds)}f", *binds)))}
        every = list(range(len(names)))
        identity = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] * len(names)
        self.assertIsNone(engine.output_problem(vrm(every, identity)))
        self.assertIn("isn't skinned to: " + names[-1], engine.output_problem(vrm(every[:-1], identity[:-16])))
        # Tripo's rigged GLB: every bone imported at NaN
        self.assertIn("bind matrices", engine.output_problem(vrm(every, [1e31] + identity[1:])))
        self.assertIn("bind matrices", engine.output_problem(vrm(every, [float("nan")] + identity[1:])))

    def test_blender_starts_on_factory_settings_with_the_startup_script_before_the_file(self):
        command = engine._command("blender", self.root / "in.blend", ["to_tpose.py", "out.blend", "out.vrm"], ["--python-expr", "pass"])
        at = command.index
        self.assertLess(at("--factory-startup"), at(str(engine.STARTUP)))
        # the VRM Add-on has to be on before the file opens, or its VRM data doesn't read
        self.assertLess(at(str(engine.STARTUP)), at(str(self.root / "in.blend")))
        self.assertLess(at(str(self.root / "in.blend")), at("--python-expr"))
        self.assertLess(at("--python-expr"), at(str(engine.SCRIPTS / "to_tpose.py")))
        self.assertEqual(command[at("--") + 1:], ["out.blend", "out.vrm"])
        self.assertTrue(engine.STARTUP.exists())

    def hanging(self):
        """A stand-in for Popen that never exits until it is killed; counts how often it was started."""
        test = self
        test.started, test.killed = 0, False

        class Hanging:
            returncode = None

            def __init__(self, *args, **kwargs):
                test.started += 1

            def poll(self):
                return -9 if test.killed else None

            def kill(self):
                test.killed = True

            def wait(self):
                return -9
        return Hanging

    def test_a_full_drive_stops_a_step_before_blender_starts(self):
        project = self.project()
        real = engine.LOCK, engine.subprocess.Popen
        engine.LOCK, engine.subprocess.Popen = self.root / ".blender.lock", self.hanging()
        engine.free_disk = lambda path: 10 << 20
        try:
            with self.assertRaises(engine.StepFailed) as failed:
                project.run(only="survey")
        finally:
            engine.LOCK, engine.subprocess.Popen = real
        self.assertEqual(failed.exception.known["code"], "VM-DISK-FULL")
        self.assertIn("10 MB free", str(failed.exception))
        self.assertEqual(list((project.out / "runs").glob("*/*.log")), [])  # the step's Blender never started
        self.assertEqual(project.state["steps"]["survey"]["last_failure"]["known"]["code"], "VM-DISK-FULL")

    def test_a_run_that_uses_up_the_memory_or_the_drive_is_stopped_and_says_which(self):
        real = engine.subprocess.Popen, engine.time.sleep, engine.STEP_TIMEOUT
        engine.subprocess.Popen, engine.time.sleep = self.hanging(), lambda seconds: None
        try:
            for stub, code in ((("free_memory", lambda: 1 << 20), "VM-OOM"), (("free_disk", lambda path: 1 << 20), "VM-DISK-FULL"),
                               (("STEP_TIMEOUT", -1), "VM-TIMEOUT")):
                self.killed, was = False, getattr(engine, stub[0])
                setattr(engine, stub[0], stub[1])
                log = self.root / f"{code}.log"
                try:
                    self.assertEqual(engine._blender("blender", None, ["preflight.py"], log, []), engine.STOPPED)
                finally:
                    setattr(engine, stub[0], was)
                self.assertTrue(self.killed)
                self.assertEqual(errors.classify("survey", log.read_text())["code"], code)
        finally:
            engine.subprocess.Popen, engine.time.sleep, engine.STEP_TIMEOUT = real

    def test_a_step_is_back_as_soon_as_its_blender_ends(self):
        # asleep 2 s between looks, every step came back up to 2 s after its Blender had ended: some
        # fifteen seconds a build
        looks, slept = [], []

        class Quick:
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            def poll(self):
                looks.append(True)
                return 0 if len(looks) > 3 else None

        def read(log, started):
            self.fail("the log, the drive and the memory are read every 2 s, not at every look")
        real = engine.subprocess.Popen, engine.time.sleep, engine._stop_reason
        engine.subprocess.Popen, engine.time.sleep, engine._stop_reason = Quick, slept.append, read
        try:
            self.assertEqual(engine._blender("blender", None, ["preflight.py"], self.root / "quick.log", []), 0)
        finally:
            engine.subprocess.Popen, engine.time.sleep, engine._stop_reason = real
        self.assertEqual(slept, [engine.POLL] * 3)
        self.assertLessEqual(engine.POLL, 0.25)
        self.assertAlmostEqual(engine.POLL * engine.WATCH, 2.0)

    def test_an_optional_step_the_engine_stopped_is_a_failure_not_nothing_to_do(self):
        project = self.project({"options": {}, "triangles": 40000})
        built = self.root / "build.blend"
        built.write_text("")
        project._record("build", {"status": "done", "blend": str(built)})
        real = engine.LOCK, engine.subprocess.Popen, engine.time.sleep
        engine.LOCK, engine.subprocess.Popen, engine.time.sleep = self.root / ".blender.lock", self.hanging(), lambda seconds: None
        engine.free_memory = lambda: 1 << 20
        try:
            with self.assertRaises(engine.StepFailed) as failed:
                project.run(only="loose")
        finally:
            engine.LOCK, engine.subprocess.Popen, engine.time.sleep = real
        self.assertEqual(failed.exception.known["code"], "VM-OOM")
        self.assertNotEqual(project.state["steps"].get("loose", {}).get("status"), "nothing to do")

    def test_a_run_starts_by_saying_what_is_short(self):
        self.assertEqual(engine.resource_warnings(self.root), [])
        engine.free_memory, engine.free_disk = lambda: 1 << 30, lambda path: 2 << 30
        notes = engine.resource_warnings(self.root)
        self.assertTrue(any("1.0 GB of memory" in n for n in notes), notes)
        self.assertTrue(any("2.0 GB free" in n and "free up space" in n for n in notes), notes)

    def write_glb(self, name, gltf, binary=b""):
        text = json.dumps(gltf).encode()
        text += b" " * (-len(text) % 4)
        binary += b"\0" * (-len(binary) % 4)
        chunks = struct.pack("<I", len(text)) + b"JSON" + text + (struct.pack("<I", len(binary)) + b"BIN\0" + binary if binary else b"")
        path = self.root / name
        path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks)
        return path

    def avatar(self, turned=None, arm=(0.2, 0.0, 0.0), look="expression", blink=True, vowels=engine.VOWELS, binds=None):
        """A small VRM's JSON: hips, both arms out along `arm` (mirrored), a face's expressions."""
        names = ["Hips", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand", "LeftIndexProximal"]
        mirrored = [-arm[0], arm[1], arm[2]]
        nodes = [{"name": "Hips", "translation": [0, 1, 0], "children": [1, 4]},
                 {"name": names[1], "translation": [0.1, 0.4, 0], "children": [2]}, {"name": names[2], "translation": list(arm), "children": [3]},
                 {"name": names[3], "translation": list(arm), "children": [7]},
                 {"name": names[4], "translation": [-0.1, 0.4, 0], "children": [5]}, {"name": names[5], "translation": mirrored, "children": [6]},
                 {"name": names[6], "translation": mirrored}, {"name": names[7], "translation": [0.05, 0, 0]},
                 {"name": "Body", "mesh": 0, "skin": 0}]
        if turned is not None:
            nodes[turned]["rotation"] = [0.0, 0.0, 0.7071, 0.7071]
        bound = [{"node": 8, "index": 0, "weight": 1.0}]
        presets = {name: {"morphTargetBinds": bound} for name in [*vowels, *(engine.LOOKS if look == "expression" else ()),
                                                                 *(["blink"] if blink else [])]}
        presets["happy"] = {}
        bones = {key: {"node": names.index(key[0].upper() + key[1:])} for key in
                 ("hips", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand", "leftIndexProximal")}
        return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(8)), **(binds or {})}],
                "meshes": [{"primitives": [], "extras": {"targetNames": ["aa", "eyeBlinkLeft"]}}],
                "materials": [{"extensions": {"VRMC_materials_mtoon": {}}}, {}], "images": [{}],
                "extensions": {"VRMC_vrm": {"specVersion": "1.0", "humanoid": {"humanBones": bones},
                                            "expressions": {"preset": presets}, "lookAt": {"type": look}},
                               "VRMC_springBone": {"springs": [{}, {}], "colliders": [{}]}}}

    def test_what_ships_is_read_from_the_exported_file(self):
        ships = engine.shipped(self.write_glb("good.vrm", self.avatar()))
        self.assertEqual((ships["bones"], ships["fingers"], ships["turned_joints"], ships["arms_off_level"]), (8, 1, [], 0.0))
        self.assertEqual((ships["blink"], ships["vowels"], ships["look"], ships["looks_bound"]), (True, list(engine.VOWELS), "expression", True))
        self.assertEqual((ships["springs"], ships["colliders"], ships["materials"], ships["toon"], ships["textures"]), (2, 1, 2, 1, 1))
        self.assertEqual(ships["shapes"], ["aa", "eyeBlinkLeft"])
        everything = {"tpose": True, "eyes": True, "blink": True}
        self.assertEqual(engine.shipping_problems(ships, everything), ([], []))
        self.assertIsNone(engine.shipped(self.write_glb("plain.glb", {"asset": {}, "nodes": []})))

    def test_a_file_that_would_track_wrongly_is_said_to(self):
        everything = {"tpose": True, "eyes": True, "blink": True}
        problems = lambda **avatar: engine.shipping_problems(engine.shipped(self.write_glb("a.vrm", self.avatar(**avatar))), everything)  # noqa: E731
        issues, looks = problems(turned=7)  # a finger left with a bone frame: Warudo twists it
        self.assertTrue(len(issues) == 1 and "1 joints rest rotated" in issues[0] and "LeftIndexProximal" in issues[0], issues)
        issues, _ = problems(arm=(0.07, -0.19, 0.0))  # an A-pose shipped as the rest pose
        self.assertTrue(len(issues) == 1 and "isn't a T-pose (arms 69.8 degrees off level)" in issues[0], issues)
        ships = engine.shipped(self.write_glb("a.vrm", self.avatar(arm=(0.07, -0.19, 0.0))))
        self.assertEqual(engine.shipping_problems(ships, {"tpose": False})[0], [])  # no T-pose was asked for
        issues, looks = problems(look="bone", blink=False, vowels=("aa", "ih", "ou"))
        self.assertEqual(len(issues), 1)
        self.assertIn("eyes aren't bound to the gaze", issues[0])
        self.assertEqual(looks, ["the exported file has no blink, though the face was built to have one",
                                 "the exported file has no mouth shape for ee, oh: the avatar can't say them"])
        self.assertEqual(engine.shipping_problems(engine.shipped(self.write_glb("a.vrm", self.avatar(look="bone", blink=False))), {}), ([], []))

    def test_the_verdict_takes_the_exported_file_in(self):
        project = self.project({"options": {}, "triangles": 40000})
        vrm = self.write_glb("09_tpose.vrm", self.avatar(turned=3))
        self.finished(project, tpose={"status": "done", "vrm": str(vrm)})
        verdict = project.verdict([])
        self.assertFalse(verdict["ready"])
        self.assertTrue(any(page == "build" and text.startswith("exported file: 1 joints rest rotated") for page, text in verdict["issues"]))
        self.assertEqual(project.status()["ships"]["turned_joints"], ["LeftHand"])
        # kept with the check when it ran: an older build is read from its file instead
        project.state["steps"]["vrm_check"]["report"] = {"ships": engine.shipped(self.write_glb("good.vrm", self.avatar()))}
        self.assertTrue(project.verdict([])["ready"])

    def test_getting_ready_does_what_is_missing_and_surveys_the_patched_model(self):
        calls, patched = [], []

        def run(project, only=None, on_event=None, **rest):
            calls.append(only)
            report = project.out / f"survey_{len(calls)}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"options": {}, "triangles": 40000, "face_patch": bool(patched)}))
            project._record("survey", {"status": "done", "json": str(report)})

        def pieces(args):
            calls.append("pieces")
            (Project(args.manifest).out / "pieces.json").write_text("{}")

        def face_patch(args):
            calls.append("face patch")
            patched.append(True)
        real = Project.run, cli.cmd_pieces, cli.cmd_face_patch
        Project.run, cli.cmd_pieces, cli.cmd_face_patch = run, pieces, face_patch
        try:
            events = []
            done = cli.prepare(self.path, on_event=events.append)
            self.assertEqual(done["did"], ["survey", "pieces", "face patch"])
            # the patched copy is the model the build reads: the survey has to be of it
            self.assertEqual(calls, ["survey", "pieces", "face patch", "survey"])
            self.assertEqual([e["step"] for e in events if e["status"] == "running"], ["pieces", "face patch"])
            self.assertEqual(cli.prepare(self.path)["did"], [])
            self.assertEqual(len(calls), 4)
        finally:
            Project.run, cli.cmd_pieces, cli.cmd_face_patch = real

    def test_a_regression_run_rebuilds_in_a_scratch_folder_and_compares_with_the_baseline(self):
        project = self.project({"options": {}, "triangles": 40000})
        check = lambda head: {"poses": {"head turn": {"stretch_cm": head}}, "unweighted": 0, "worst": {}, "failed": []}  # noqa: E731
        args = type("A", (), {"manifests": [self.path], "keep": False, "blender": "blender"})()
        real = Project.run, cli.REPO
        cli.REPO = self.root
        try:
            self.assertIn("never built", cli.cmd_regress(args)["characters"][0]["error"])
            base = self.root / "base_check.json"
            base.write_text(json.dumps(check(2.6)))
            project._record("check", {"status": "done", "json": str(base)})
            (project.out / "pieces.json").write_text("{}")

            def run(scratch, on_event=None, **rest):
                folder = scratch.out / "runs" / "20260101-100000"
                folder.mkdir(parents=True)
                for name in ("09_tpose.blend", "09_tpose.vrm"):
                    (folder / name).write_text("x")
                (folder / "12_check.json").write_text(json.dumps(check(8.0)))
                scratch._record("check", {"status": "done", "json": str(folder / "12_check.json")})
            Project.run = run
            result = cli.cmd_regress(args)
        finally:
            Project.run, cli.REPO = real
        row = result["characters"][0]
        self.assertEqual((row["baseline"], row["error"], result["worse"]), ("the last build", None, 1))
        self.assertEqual(row["worse"], ["rig check, head turn: stretch 2.6 -> 8.0 cm"])
        scratch = self.root / "out" / "regress" / "Test"
        copy = json.loads((scratch / "character.json").read_text())
        # the copy lives elsewhere: its paths are absolute and its builds go to its own folder
        self.assertEqual((copy["source"], copy["out_dir"]), (str((self.root / "model.glb").resolve()), str(scratch)))
        self.assertTrue((scratch / "pieces.json").exists())
        self.assertEqual(sorted(p.name for p in (scratch / "runs" / "20260101-100000").iterdir()), ["12_check.json"])
        # the character itself is as it was
        self.assertEqual(Project(self.path).state["steps"]["check"]["json"], str(base))

    def test_a_regression_run_forgets_the_eyes_an_older_run_fell_back_to(self):
        # the scratch folder outlives the pipeline that refused the iris there
        project = self.project({"options": {}, "triangles": 40000})
        real, cli.REPO = cli.REPO, self.root
        try:
            first = cli._scratch_copy(project)
            first.state["eyes_fallback"] = "slide"
            first._save_state()
            self.assertNotIn("eyes_fallback", cli._scratch_copy(project).state)
            self.assertNotIn("eyes_fallback", Project(first.out / "character.json").state)
        finally:
            cli.REPO = real

    def test_old_runs_give_up_their_models_and_keep_their_reports(self):
        project = self.project()
        runs = project.out / "runs"
        made = {}
        for run, files in (("20260101-100000", ["01_build.blend", "01_build.vrm", "01_build.log", "12_check.json"]),   # old
                           ("20260102-100000", ["09_tpose.blend", "09_tpose.vrm", "09_tpose.blend1"]),                 # marked good
                           ("20260103-100000", ["09_tpose.blend", "09_tpose.vrm"]),                                    # has a note
                           ("20260104-100000", ["09_tpose.blend", "09_tpose.vrm", "09_tpose.log"])):                   # in use
            (runs / run).mkdir(parents=True)
            for name in files:
                made[run, name] = runs / run / name
                made[run, name].write_bytes(b"x" * 1000)
        project.state["steps"]["tpose"] = {"status": "done", "blend": str(made["20260104-100000", "09_tpose.blend"]),
                                           "vrm": str(made["20260104-100000", "09_tpose.vrm"])}
        (project.out / "approved").mkdir()
        (project.out / "approved" / "approved.json").write_text(json.dumps({"vrm": str(made["20260102-100000", "09_tpose.vrm"])}))
        (project.out / "feedback" / "a").mkdir(parents=True)
        (project.out / "feedback" / "a" / "feedback.json").write_text(json.dumps({"vrm": str(made["20260103-100000", "09_tpose.vrm"])}))
        usage = project.usage()
        self.assertEqual([(r["run"], r["kept"], r["models"]) for r in usage["runs"]],
                         [("20260104-100000", True, 2000), ("20260103-100000", True, 2000), ("20260102-100000", True, 3000),
                          ("20260101-100000", False, 2000)])
        self.assertEqual((usage["prunable"], usage["prunable_runs"]), (2000, 1))
        self.assertEqual(usage["bytes"], sum(p.stat().st_size for p in project.out.rglob("*") if p.is_file()))
        self.assertEqual(project.prune(dry=True)["freed"], 2000)
        self.assertTrue(made["20260101-100000", "01_build.blend"].exists())  # a dry run only counts
        real, engine.LOCK = engine.LOCK, self.root / ".blender.lock"
        try:
            with engine._Lock():  # a build is running: its folder isn't in the state yet
                with self.assertRaisesRegex(engine.StepFailed, "another build is running"):
                    project.prune()
            self.assertEqual(project.prune(), {"name": "Test", "removed": 2, "freed": 2000, "dry": False})
        finally:
            engine.LOCK = real
        left = sorted(p.name for p in (runs / "20260101-100000").iterdir())
        self.assertEqual(left, ["01_build.log", "12_check.json"])
        self.assertTrue(all(path.exists() for (run, _), path in made.items() if run != "20260101-100000"))
        self.assertEqual(project.usage()["prunable"], 0)

    def test_a_file_that_goes_while_the_disk_is_measured_counts_for_nothing(self):
        # a build renames its temporary files and its browser empties a profile inside the run's folder: one
        # gone between being listed and being measured ended the whole count
        folder = self.root / "tree"
        (folder / "run" / "profile").mkdir(parents=True)
        (folder / "a.blend").write_bytes(b"x" * 300)
        (folder / "run" / "profile" / "cache").write_bytes(b"x" * 50)
        self.assertEqual(engine.tree_size(folder), 350)
        self.assertEqual(engine.tree_size(self.root / "no such folder"), 0)
        real = os.path.getsize

        def gone(path):
            if str(path).endswith("cache"):
                raise FileNotFoundError(path)
            return real(path)
        os.path.getsize = gone
        try:
            self.assertEqual(engine.tree_size(folder), 300)
        finally:
            os.path.getsize = real

    def test_the_finished_vrm_is_sent_to_its_folder_once_and_never_written_over(self):
        project = self.project()
        library = self.root / "SteamLibrary"
        real, engine.steam_libraries = engine.steam_libraries, lambda: [self.root / "Steam", library]
        try:
            vrm = project.out / "runs" / "20260104-100000" / "13_export.vrm"
            vrm.parent.mkdir(parents=True)
            vrm.write_bytes(b"x" * 1000)
            with self.assertRaisesRegex(engine.StepFailed, "build first"):
                project.send_to_folder()
            project.state["steps"]["export"] = {"status": "done", "vrm": str(vrm)}
            with self.assertRaisesRegex(FileNotFoundError, "no folder to send avatars to"):
                project.send_to_folder()
            characters = library / engine.WARUDO_CHARACTERS
            characters.mkdir(parents=True)
            sent = project.send_to_folder()
            self.assertEqual((sent["name"], sent["already"]), ("Test_20260104-100000.vrm", False))
            self.assertEqual([p.name for p in characters.iterdir()], ["Test_20260104-100000.vrm"])
            self.assertTrue(project.send_to_folder()["already"])
            (characters / sent["name"]).write_bytes(b"someone else's")
            with self.assertRaisesRegex(engine.StepFailed, "a different file"):
                project.send_to_folder()
            self.assertEqual((characters / sent["name"]).read_bytes(), b"someone else's")
            # a folder picked by hand wins over Steam's Warudo, and only a folder is taken; only a picked one
            # counts as chosen (the page asks for one before the first send)
            self.assertIsNone(engine.send_folder(chosen=True))
            mine = self.root / "VNyan" / "Avatars"
            with self.assertRaisesRegex(ValueError, "isn't a folder"):
                engine.use_folder(str(mine))
            mine.mkdir(parents=True)
            self.assertEqual(engine.use_folder(str(mine)), {"send_to": str(mine)})
            self.assertEqual(Path(project.send_to_folder()["sent"]).parent, mine)
            self.assertEqual(engine.use_folder(""), {"send_to": ""})
            self.assertEqual(engine.send_folder(), characters)
        finally:
            engine.steam_libraries = real

    def test_the_tested_blender_is_picked_before_a_newer_untested_one(self):
        folders = ["Blender 4.2", "Blender 4.10", "blender-5.1.0-windows-x64", "Blender 3.6"]
        paths = [self.root / f / "blender.exe" for f in folders]
        for path in paths:
            path.parent.mkdir()
            path.write_text("")
        real = engine.installed_blenders, engine.shutil.which, engine.os.environ.pop("BLENDER", None)
        real_tested = set(engine.TESTED)
        engine.TESTED.clear()
        engine.TESTED.add("4.2")
        engine.installed_blenders, engine.shutil.which = lambda: paths, lambda name: None
        try:
            self.assertEqual(engine.find_blender(), str(paths[0]))
            engine.TESTED.add("4.10")  # parsed as numbers: 4.10 is newer than 4.2
            self.assertEqual(engine.find_blender(), str(paths[1]))
            engine.TESTED.clear()  # nothing tested: the newest, never one below 4.2
            self.assertEqual(engine.find_blender(), str(paths[2]))
        finally:
            engine.TESTED.clear()
            engine.TESTED.update(real_tested)
            engine.installed_blenders, engine.shutil.which = real[:2]
            if real[2]:
                engine.os.environ["BLENDER"] = real[2]

    def no_blender_but(self, *found):
        """This machine with only the given blender.exe paths on it (each a (path, version) pair),
        and every Python package the app uses: the tests run in Blender's Python too, which has no Pillow."""
        real = (engine.installed_blenders, engine.shutil.which, engine.os.environ.pop("BLENDER", None),
                engine.os.environ.get("APPDATA"), dict(engine._BLENDER_VERSIONS), engine._installed, engine.portable_blenders)
        engine.installed_blenders, engine.shutil.which = lambda: [path for path, _ in found], lambda name: None
        engine._installed, engine.portable_blenders = lambda module: True, lambda: []
        engine.os.environ["APPDATA"] = str(self.root / "appdata")
        for path, version in found:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            engine._BLENDER_VERSIONS[str(path)] = version

        def restore():
            engine.installed_blenders, engine.shutil.which, engine._installed, engine.portable_blenders = real[0], real[1], real[5], real[6]
            for key, value in (("BLENDER", real[2]), ("APPDATA", real[3])):
                engine.os.environ.pop(key, None)
                if value:
                    engine.os.environ[key] = value
            engine._BLENDER_VERSIONS.clear()
            engine._BLENDER_VERSIONS.update(real[4])
        self.addCleanup(restore)

    def test_the_blender_chosen_on_the_setup_page_is_the_one_run(self):
        installed, chosen = self.root / "Blender 4.2" / "blender.exe", self.root / "portable" / "blender-5.2.2-windows-x64" / "blender.exe"
        self.no_blender_but((installed, "4.2.1"))
        chosen.parent.mkdir(parents=True)
        chosen.write_text("")
        engine._BLENDER_VERSIONS[str(chosen)] = "5.2.2"
        self.assertEqual(engine.find_blender(), str(installed))
        engine.use_blender(str(chosen))
        self.assertEqual(engine.find_blender(), str(chosen))
        self.assertEqual(engine.find_blender(str(installed)), str(installed))  # --blender still wins
        old = self.root / "Blender 3.6" / "blender.exe"
        old.parent.mkdir()
        old.write_text("")
        engine._BLENDER_VERSIONS[str(old)] = "3.6.0"
        with self.assertRaisesRegex(ValueError, r"isn't a Blender 4.2 or later \(it is 3.6.0\)"):
            engine.use_blender(str(old))
        engine.use_blender("")
        self.assertEqual(engine.find_blender(), str(installed))

    def test_a_blender_the_app_didnt_find_can_be_added_and_taken_off_again(self):
        installed, mine = self.root / "Blender 4.2" / "blender.exe", self.root / "Apps" / "my build" / "blender.exe"
        self.no_blender_but((installed, "4.2.1"))
        mine.parent.mkdir(parents=True)
        mine.write_text("")
        engine._BLENDER_VERSIONS[str(mine)] = "5.2.2"   # its folder says nothing: it is asked
        report = engine.use_blender(str(mine))
        self.assertEqual((report["blender"], report["version"], report["chosen"]), (str(mine), "5.2.2", str(mine)))
        rows = {b["path"]: b for b in report["blenders"]}
        self.assertEqual((rows[str(mine)]["added"], rows[str(mine)]["version"], rows[str(mine)]["missing"]), (True, "5.2.2", False))
        self.assertFalse(rows[str(installed)]["added"])
        self.assertEqual([b["path"] for b in report["blenders"]][0], str(mine))  # newest first
        engine.use_blender(str(mine))  # chosen twice: listed once
        self.assertEqual(sum(b["added"] for b in engine.known_blenders()), 1)
        # gone from the disk: said so, and builds fall back to one the app finds
        mine.unlink()
        gone = engine.setup()
        self.assertTrue(next(b for b in gone["blenders"] if b["added"])["missing"])
        self.assertEqual(gone["blender"], str(installed))
        self.assertIn("is gone", next(c for c in gone["checks"] if c["name"] == "Chosen Blender")["text"])
        after = engine.forget_blender(str(mine))
        self.assertEqual((after["chosen"], [b["added"] for b in after["blenders"]]), (None, [False]))

    def test_only_a_file_named_blender_is_ever_started(self):
        # the path comes from a file dialog: an installer picked by mistake must not be run to ask its version
        other = self.root / "Downloads" / "setup.exe"
        self.no_blender_but()
        other.parent.mkdir()
        other.write_text("")
        real = engine.subprocess.run
        started = []
        engine.subprocess.run = lambda command, **rest: started.append(command)
        try:
            with self.assertRaisesRegex(ValueError, "isn't a blender.exe"):
                engine.use_blender(str(other))
            with self.assertRaisesRegex(ValueError, "isn't a blender.exe"):
                engine.use_blender(str(self.root / "nowhere" / "blender.exe"))
        finally:
            engine.subprocess.run = real
        self.assertEqual(started, [])

    def test_a_blender_that_dies_swapping_memory_routines_is_asked_again(self):
        # Blender 5.2.2 and 4.5.14 after the Windows updates of late October 2026: "Failed to replace function
        # free in module ucrtbase.dll" and 0xC0000409 before an argument was read; both run with TBB's switch set
        real = engine.subprocess.run, dict(engine._BLENDER_VERSIONS), os.environ.pop(engine.MALLOC_SWITCH, None)
        asked = []

        def run(command, **rest):
            asked.append(engine.MALLOC_SWITCH in os.environ)
            if asked[-1]:
                return SimpleNamespace(stdout="Blender 5.2.2 LTS\n", stderr="", returncode=0)
            return SimpleNamespace(stdout="", stderr="Failed to replace function free in module ucrtbase.dll\n", returncode=3221226505)
        engine.subprocess.run = run
        try:
            self.assertEqual(engine.blender_version(self.root / "new" / "blender.exe"), "5.2.2")
            self.assertEqual(asked, [False, True])
            self.assertEqual(os.environ.get(engine.MALLOC_SWITCH), "1")
            # ... and one that says nothing with the switch either is no reason to leave it set
            del os.environ[engine.MALLOC_SWITCH]
            engine.subprocess.run = lambda command, **rest: SimpleNamespace(stdout="", stderr="Failed to replace function free", returncode=1)
            self.assertIsNone(engine.blender_version(self.root / "dead" / "blender.exe"))
            self.assertNotIn(engine.MALLOC_SWITCH, os.environ)
        finally:
            engine.subprocess.run = real[0]
            engine._BLENDER_VERSIONS.clear()
            engine._BLENDER_VERSIONS.update(real[1])
            os.environ.pop(engine.MALLOC_SWITCH, None)
            if real[2] is not None:
                os.environ[engine.MALLOC_SWITCH] = real[2]

    def test_setup_says_what_is_missing_and_what_to_do(self):
        self.no_blender_but()
        found = {c["name"]: c for c in engine.setup()["checks"]}
        self.assertEqual(found["Blender"]["state"], "bad")
        self.assertIn("blender.org", found["Blender"]["fix"])
        self.assertFalse(engine.setup()["ready"])
        blender = self.root / "blender-5.2.2-windows-x64" / "blender.exe"
        self.no_blender_but((blender, "5.2.2"))
        report = engine.setup()
        found = {c["name"]: c for c in report["checks"]}
        self.assertEqual((found["Blender"]["state"], found["VRM Add-on"]["state"], report["ready"]), ("ok", "bad", False))
        self.assertIn("Get Extensions", found["VRM Add-on"]["fix"])
        addon = blender.parent / "portable" / "extensions" / "user_default" / "vrm" / "blender_manifest.toml"
        addon.parent.mkdir(parents=True)
        addon.write_text('id = "vrm"\nversion = "4.7.2"\n')
        report = engine.setup()
        self.assertEqual(next(c for c in report["checks"] if c["name"] == "VRM Add-on")["text"], "VRM Add-on 4.7.2")
        self.assertTrue(report["ready"])
        engine._installed = lambda module: module != "PIL"  # no Pillow: the face patch can't be made
        missing = engine.setup()
        self.assertEqual([(c["name"], c["fix"]) for c in missing["checks"] if c["state"] == "bad"], [("Pictures", "pip install pillow")])
        self.assertFalse(missing["ready"])
        engine._installed = lambda module: True
        # a series past its support still builds, and says so
        old = self.root / "Blender 4.2" / "blender.exe"
        self.no_blender_but((old, "4.2.1"))
        aged = next(c for c in engine.setup()["checks"] if c["name"] == "Blender")
        self.assertEqual(aged["state"], "warn")
        self.assertIn("stopped getting fixes in 2026-07", aged["text"])

    def test_an_older_manifest_is_carried_forward_and_a_newer_one_refused(self):
        real = manifest.VERSION, dict(manifest.MIGRATIONS)
        try:
            manifest.VERSION = 2
            manifest.MIGRATIONS[1] = lambda data: {**data, "version": 2, "carried": True}
            self.assertTrue(manifest.migrate(json.loads(self.path.read_text()))["carried"])
            with self.assertRaisesRegex(ValueError, "newer version"):
                manifest.migrate({"version": 3})
        finally:
            manifest.VERSION = real[0]
            manifest.MIGRATIONS.clear()
            manifest.MIGRATIONS.update(real[1])


class NoFingers(unittest.TestCase):
    """A model with mitten hands still builds: the steps that work on fingers stand aside."""

    def test_the_guides_step_is_skipped_when_no_fingers_were_found(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character.json"
            data = manifest.new("model.glb", "Test")
            data["out_dir"] = "out"
            manifest.save(data, path)
            project = Project(path, blender="blender")
            plan = {name: (runs, why) for name, runs, why in project.plan()}
            self.assertTrue(plan["finger_guides"][0])
            project.state["steps"]["fingers"] = {"status": "nothing to do"}
            plan = {name: (runs, why) for name, runs, why in project.plan()}
            self.assertEqual(plan["finger_guides"], (False, "no separate fingers were found on this model"))
            self.assertTrue(plan["tpose"][0] and plan["export"][0])
            project.state["steps"]["fingers"] = {"status": "done"}
            self.assertTrue(dict((n, r) for n, r, _ in project.plan())["finger_guides"])


class ModelWithItsOwnRig(unittest.TestCase):
    """A character whose source is an MMD model as FBX is taken in with its rig, not rigged again."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        (self.root / "game.fbx").write_bytes(b"Kaydara FBX Binary\0Model::Left elbow\0\1LimbNode")
        (self.root / "tripo.fbx").write_bytes(b"Kaydara FBX Binary\0Model::J_Bip_C_Hips")
        self.path = self.root / "character.json"
        self.real_settings, engine.SETTINGS = engine.SETTINGS, self.root / "settings.json"

    def tearDown(self):
        engine.SETTINGS = self.real_settings
        self.folder.cleanup()

    def project(self, source="game.fbx", **options):
        data = manifest.new(source, "Test")
        data["out_dir"] = "out"
        data["options"].update(options)
        manifest.save(data, self.path)
        return Project(self.path, blender="blender")

    def test_its_rig_is_kept_and_only_the_pose_and_the_checks_run(self):
        project = self.project(chest=True)
        self.assertTrue(project.keeping and project.improving)
        self.assertFalse(manifest.improves(project.data))
        plan = {name: (runs, why) for name, runs, why in project.plan()}
        self.assertEqual({name for name, (runs, _) in plan.items() if runs},
                         {"build", "tpose", "check", "export", "vrm_check", "springs"})
        self.assertEqual(plan["hair"], (False, "the model's own hair weights and springs stay"))
        self.assertEqual(plan["chest"][0], False)
        build, extra = project._arguments("build", self.root / "out" / "01_build", None)
        self.assertEqual(build[0], "import_rigged.py")
        self.assertEqual(extra, engine.NO_EXPORT)
        pose, _ = project._arguments("tpose", self.root / "out" / "09_tpose", None)
        self.assertIn("--keep-finger-pose", pose)
        self.assertNotIn("--spec-hands", pose)

    def test_the_user_can_have_it_built_as_a_bare_surface_and_other_rigs_are_not_kept(self):
        self.assertFalse(self.project(keep_rig=False).improving)
        # Tripo's rig is a body's alone: adopted (the manifest's `rig`), with the face and hair made here
        self.assertFalse(self.project(source="tripo.fbx").keeping)
        self.assertEqual(manifest.inert(self.project(skinning="voxel", hair_hits_body=False).data), ["skinning"])


class FinishedVrm(unittest.TestCase):
    """A character whose source is a finished .vrm is taken in as it is and added to, not built."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = Engine.write_glb(self, "avatar.vrm", Engine.avatar(self, turned=3))
        self.path = self.root / "character.json"
        data = manifest.new("avatar.vrm", "Test")
        data["out_dir"] = "out"
        manifest.save(data, self.path)
        self.real_settings, engine.SETTINGS = engine.SETTINGS, self.root / "settings.json"

    def tearDown(self):
        engine.SETTINGS = self.real_settings
        self.folder.cleanup()

    def project(self, **options):
        data = manifest.load(self.path)
        data["options"].update(options)
        manifest.save(data, self.path)
        return Project(self.path, blender="blender")

    def finished(self, project, ships):
        vrm = Engine.write_glb(self, "14_export.vrm", Engine.avatar(self))
        project.state["steps"]["export"] = {"status": "done", "vrm": str(vrm)}
        project.state["steps"]["vrm_check"] = {"status": "done", "report": {"ships": ships}}

    def test_only_what_adds_to_the_file_runs(self):
        project = self.project()
        self.assertTrue(project.improving)
        self.assertFalse(manifest.improves(manifest.new("model.glb", "Other")))
        plan = {name: (runs, why) for name, runs, why in project.plan()}
        self.assertEqual({name for name, (runs, _) in plan.items() if runs}, {"build", "check", "export", "vrm_check", "springs"})
        self.assertEqual(plan["survey"], (False, "a finished .vrm is read as it is, not surveyed"))
        self.assertEqual(plan["fingers"], (False, "the file's own fingers stay"))
        self.assertIn("chest", {name for name, runs, _ in self.project(chest=True).plan() if runs})

    def test_the_file_is_read_not_rigged_and_checked_whole(self):
        project = self.project()
        stem = self.root / "out" / "runs" / "now" / "01_build"
        args, extra = project._arguments("build", stem, None)
        self.assertEqual((args[0], Path(args[1])), ("import_vrm.py", self.source.resolve()))
        self.assertEqual(extra, engine.NO_EXPORT)
        _, extra = project._arguments("check", stem.with_name("13_check"), stem.with_suffix(".blend"))
        self.assertEqual(extra, engine.WHOLE_BODY)
        self.finished(project, None)
        args, extra = project._arguments("vrm_check", stem.with_name("15_vrm_check"), None)
        # the exported file has its own bone names back: read through the renaming, its meshes joined
        self.assertEqual(Path(args[1]).name, "14_export.vrm")
        self.assertIn(str(engine.SCRIPTS / "load_vrm_as_ours.py"), extra)
        self.assertEqual(extra[-len(engine.WHOLE_BODY):], engine.WHOLE_BODY)

    def test_the_height_that_scales_things_is_the_files_own(self):
        project = self.project()
        self.assertEqual(project.options()["height"], 1.6)
        project.state["steps"]["build"] = {"status": "done", "report": {"height": 1.42}}
        self.assertEqual(project.options()["height"], 1.42)

    def test_the_status_says_what_the_file_came_with_and_what_it_gained(self):
        project = self.project()
        came = project.source_holds()
        self.assertEqual((came["bones"], came["turned_joints"], came["springs"]), (8, ["LeftHand"], 2))
        self.assertEqual(project.status()["gained"], [])  # nothing built yet
        self.finished(project, dict(engine.shipped(Engine.write_glb(self, "out.vrm", Engine.avatar(self))), springs=4))
        gained = project.status()["gained"]
        self.assertEqual(len(gained), 2)
        self.assertTrue(gained[0].startswith("1 joints rested turned in the file, none now"))
        self.assertEqual(gained[1], "2 springs added (2 before, 4 now)")
        self.assertEqual(project.verdict([]), {"ready": True, "issues": [], "looks": []})

    def test_anything_the_file_had_and_the_export_lacks_blocks(self):
        project = self.project()
        whole = engine.shipped(Engine.write_glb(self, "out.vrm", Engine.avatar(self)))
        self.assertEqual(engine.lost(project.source_holds(), whole), [])
        self.assertEqual(engine.lost(None, whole), [])  # a character built from a model has no file to keep
        less = dict(whole, springs=1, blink=False, vowels=["aa", "ih", "ou", "ee"], shapes=["aa"], look="bone", toon=0)
        self.assertEqual(engine.lost(project.source_holds(), less), [
            "1 springs, the file taken in had 2", "0 toon materials, the file taken in had 1",
            "1 face shapes of the file taken in are gone (eyeBlinkLeft)", "the file taken in could blink; this one can't",
            "the file taken in had mouth shapes for oh; this one doesn't",
            "the eyes no longer follow the gaze the way they did in the file taken in"])
        self.finished(project, less)
        verdict = project.verdict([])
        self.assertFalse(verdict["ready"])
        self.assertIn(("build", "exported file: 1 springs, the file taken in had 2"), verdict["issues"])

    def test_the_files_own_skinning_is_said_not_held_against_the_build(self):
        project = self.project()
        self.finished(project, engine.shipped(Engine.write_glb(self, "out.vrm", Engine.avatar(self))))
        # without the check's guess at why, which is about a rig made here
        failed = ["an edge shrinks to 18% with the leg lift: a limb's weights reach onto the body it folds into"]
        verdict = project.verdict(failed)
        self.assertTrue(verdict["ready"])
        self.assertEqual(verdict["issues"], [])
        said = "rig check: an edge shrinks to 18% with the leg lift (the file's own skinning, kept as it came)"
        self.assertEqual(verdict["looks"], [("review", said)])
        # and once in a report's first lines, not as the rig check and again as a look
        from app import report
        lines = report.headline({"check_failed": failed, "verdict": {"looks": [["review", said], ["face", "the eyes: " + "long " * 60]]},
                                 "character": {"options_changed": {"skinning": "voxel"}, "inert": ["skinning"]}}, "Test")
        self.assertEqual(sum("18%" in line for line in lines), 1)
        self.assertTrue(any(line.startswith("Also flagged: the eyes") and line.endswith("long...") for line in lines))
        self.assertIn("Options changed: skinning=voxel (no effect on a finished .vrm: skinning)", lines)

    def test_a_joint_a_constraint_holds_turned_is_not_a_fault(self):
        avatar = Engine.avatar(self, turned=3)
        avatar["nodes"][3]["extensions"] = {"VRMC_node_constraint": {"constraint": {"aim": {"source": 2, "aimAxis": "PositiveX"}}}}
        # the finger under it is turned back against it: on the model's axes, so no fault either
        avatar["nodes"][3]["rotation"], avatar["nodes"][7]["rotation"] = [0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, -0.70710678, 0.70710678]
        self.assertEqual(engine.shipped(Engine.write_glb(self, "held.vrm", avatar))["turned_joints"], [])
        avatar["nodes"][7]["rotation"] = [0.0, 0.3, 0.0, 0.954]
        self.assertEqual(engine.shipped(Engine.write_glb(self, "held.vrm", avatar))["turned_joints"], ["LeftIndexProximal"])

    def test_nothing_is_got_ready_and_a_file_that_is_no_vrm_1_is_refused(self):
        self.assertEqual(cli.prepare(self.path, blender="blender")["did"], [])
        made = cli.cmd_init(SimpleNamespace(model=self.source, name="Again", author="Unknown", face=None, reference=None, rig=None,
                                            manifest=self.root / "again.json"))
        self.assertTrue(made["improving"])
        self.assertIn(" run ", made["next"])
        old = Engine.write_glb(self, "old.vrm", {"asset": {}, "nodes": [], "extensions": {"VRM": {"specVersion": "0.0"}}})
        with self.assertRaisesRegex(ValueError, "isn't a VRM file the app can read"):
            cli.cmd_init(SimpleNamespace(model=old, name="Old", author="Unknown", face=None, reference=None, rig=None,
                                         manifest=self.root / "old.json"))
        self.assertFalse((self.root / "old.json").exists())

    def test_an_option_that_changes_nothing_in_a_vrm_is_said(self):
        # a tester set skinning to voxel on a VRoid file: four identical exports and no word of it
        said = cli.cmd_set(SimpleNamespace(manifest=self.path, key="skinning", value="voxel"))
        self.assertEqual(said["options"]["skinning"], "voxel")
        self.assertIn("skinning changes nothing", said["note"])
        self.assertNotIn("note", cli.cmd_set(SimpleNamespace(manifest=self.path, key="chest", value="true")))
        self.assertEqual(manifest.inert(manifest.load(self.path)), ["skinning"])
        self.assertEqual(Project(self.path, blender="blender").status()["inert"], ["skinning"])
        model = self.root / "model.json"
        manifest.save(manifest.new("model.glb", "Other"), model)
        self.assertNotIn("note", cli.cmd_set(SimpleNamespace(manifest=model, key="skinning", value="voxel")))
        self.assertEqual(manifest.inert(manifest.load(model)), [])

    def test_the_cli_runs_under_its_module_name_too(self):
        # `python -m app.cli set ...` exited 0 having done nothing: the module had no entry of its own
        import subprocess
        asked = subprocess.run([sys.executable, "-m", "app.cli", "--help"], cwd=engine.REPO, capture_output=True, text=True)
        self.assertIn("usage: python -m app", asked.stdout)


class FirstRun(unittest.TestCase):
    """What "Get set up" stands on: the checks name what a button can do, and the installs are only the app's own."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.real_settings, engine.SETTINGS = engine.SETTINGS, Path(self.folder.name) / "settings.json"

    def tearDown(self):
        engine.SETTINGS = self.real_settings
        self.folder.cleanup()

    def test_the_app_opens_where_the_setup_page_says(self):
        real = engine._installed
        try:
            engine._installed = lambda module: True
            self.assertEqual(engine.open_in(), {"open_in": "window", "can_window": True})
            self.assertTrue(server._own_window())
            self.assertEqual(engine.open_in("browser")["open_in"], "browser")
            self.assertEqual(engine.setup()["open_in"], "browser")
            self.assertFalse(server._own_window())
            engine.open_in("window")
            # its own window is pywebview's: without it the browser opens, whatever is chosen
            engine._installed = lambda module: module != "webview"
            self.assertFalse(server._own_window())
            with self.assertRaisesRegex(ValueError, "its own window"):
                engine.open_in("a tab")
        finally:
            engine._installed = real

    def test_a_crashed_window_is_found_by_its_dump_and_its_profile_removed(self):
        import time
        temp = Path(self.folder.name)

        def profile(name, text, dump=True):
            crashpad = temp / name / "EBWebView" / "Crashpad"
            (crashpad / "reports").mkdir(parents=True)
            (crashpad / "watson_metadata").write_bytes(text.encode("latin-1"))
            if dump:
                (crashpad / "reports" / "a.dmp").write_bytes(b"MDMP")
            return temp / name

        ours = "\x00ApplicationName=msedgewebview2.exe;ModuleName=RTSSHooks64.dll;ProcessType=browser;SubCode=0xc0000005;StackHash=0;\x00WV=python.exe;"
        real_log, server.ERROR_LOG = server.ERROR_LOG, temp / "app_errors.log"
        keep, since = temp / "crashes", time.time() - 60
        try:
            self.assertIsNone(server._window_crash(since, temp, keep))
            for days in (3, 2):
                old = profile(f"tmpold{days}", ours)
                os.utime(old / "EBWebView" / "Crashpad" / "reports" / "a.dmp", (since - days * 86400,) * 2)
            open_still = profile("tmpopen", ours, dump=False)
            theirs = profile("tmptheirs", ours.replace("WV=python.exe", "WV=other.exe"))
            # a crash of an earlier day is not this window's, but it is written down all the same
            # and nothing reads its profile again
            self.assertIsNone(server._window_crash(since, temp, keep))
            self.assertFalse(old.exists())
            self.assertEqual(server.ERROR_LOG.read_text().count("crashed in RTSSHooks64.dll"), 2)
            now = profile("tmpnow", ours)
            crash = server._window_crash(since, temp, keep)
            self.assertEqual(crash["module"], "RTSSHooks64.dll")
            self.assertIn("SubCode=0xc0000005", crash["detail"])
            self.assertNotIn("\x00", crash["detail"])
            self.assertFalse(now.exists())
            self.assertTrue(open_still.exists() and theirs.exists())
            # the dumps of the last crashes are kept for whoever looks into one, the oldest let go
            dumps = sorted(p.name for p in keep.glob("*.dmp"))
            self.assertEqual(len(dumps), server.CRASHES_KEPT)
            self.assertTrue(all(name.endswith("_RTSSHooks64.dll.dmp") for name in dumps))
            self.assertIn(dumps[-1], server.ERROR_LOG.read_text())
            self.assertIn("rtsshooks64.dll", server._CRASH_HINTS)
        finally:
            server.ERROR_LOG = real_log

    def test_a_save_waits_out_a_file_held_for_a_moment_and_leaves_no_temporary(self):
        path = Path(self.folder.name) / "character.json"
        data = manifest.new("model.glb", "Held")
        manifest.save(data, path)
        real, real_wait, calls = manifest.os.replace, manifest.REPLACE_WAIT, []

        def held(source, target):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError(13, "Access is denied")
            return real(source, target)

        try:
            manifest.os.replace, manifest.REPLACE_WAIT = held, 0
            data["options"]["eyes"] = "iris"
            manifest.save(data, path)
            self.assertEqual(len(calls), 3)
            self.assertEqual(manifest.load(path)["options"]["eyes"], "iris")
            # held for good: said, and the half-made file not left beside the manifest
            calls.clear()
            manifest.os.replace = lambda source, target: (_ for _ in ()).throw(PermissionError(13, "Access is denied"))
            with self.assertRaises(PermissionError):
                manifest.save(data, path)
        finally:
            manifest.os.replace, manifest.REPLACE_WAIT = real, real_wait
        self.assertEqual([p.name for p in path.parent.glob("character.json*")], ["character.json"])

    def test_the_first_start_is_remembered_once_it_is_done(self):
        self.assertFalse(engine.setup()["welcomed"])
        self.assertEqual(engine.welcome(), {"welcomed": True})
        report = engine.setup()
        self.assertTrue(report["welcomed"])
        self.assertTrue(report["python"])
        engine.welcome(False)
        self.assertFalse(engine.setup()["welcomed"])

    def test_a_missing_blender_and_addon_name_what_gets_them(self):
        real = engine.find_blender, engine.known_blenders
        engine.known_blenders = lambda: []

        def none(given=None):
            raise FileNotFoundError("no Blender")
        try:
            engine.find_blender = none
            checks = {c["name"]: c for c in engine.setup()["checks"]}
            # on Windows the app fetches a copy for itself; elsewhere it says where Blender is got
            self.assertEqual((checks["Blender"]["state"], checks["Blender"]["action"]),
                             ("bad", "install_blender" if engine.blender_build() else "get_blender"))
            self.assertNotIn("VRM Add-on", checks)  # nothing to look in yet
        finally:
            engine.find_blender, engine.known_blenders = real

    def blender_zip(self, files):
        import hashlib
        import io
        import zipfile
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
        raw = data.getvalue()
        source = {"version": "5.2.2", "folder": "blender-test", "bytes": len(raw), "url": "https://example.invalid/blender-test.zip",
                  "sha256": hashlib.sha256(raw).hexdigest()}
        asked = []

        def open_url(request, timeout=None):
            asked.append(request.full_url)
            return io.BytesIO(raw)
        return source, open_url, asked

    def test_blender_is_fetched_into_the_apps_folder_checked_and_chosen(self):
        real = engine.REPO, engine.blender_version, engine.free_disk
        engine.REPO, engine.blender_version, engine.free_disk = Path(self.folder.name), lambda path: "5.2.2", lambda path: 100 << 30
        try:
            source, open_url, asked = self.blender_zip({"blender-test/blender.exe": "exe", "blender-test/5.2/scripts/x.py": "x"})
            events = []
            engine.install_blender(on_event=events.append, open_url=open_url, source=source, addon=False)
            home = Path(self.folder.name) / "tools" / "blender"
            exe = home / "blender-test" / "blender.exe"
            self.assertTrue(exe.is_file())
            self.assertTrue((home / "blender-test" / "portable").is_dir())  # its add-ons and settings stay with it
            self.assertEqual(sorted(p.name for p in home.iterdir()), ["blender-test"])  # no zip, no half-unzipped folder left
            self.assertEqual(Path(engine._read_json(engine.SETTINGS, {})["blender"]), exe)
            self.assertEqual(asked, [source["url"]])
            self.assertEqual([e["step"] for e in events if e["status"] == "done"], ["downloading Blender", "unzipping Blender"])
            # a copy already there is used as it is: nothing is downloaded again
            engine.install_blender(open_url=open_url, source=source, addon=False)
            self.assertEqual(len(asked), 1)
        finally:
            engine.REPO, engine.blender_version, engine.free_disk = real

    def test_a_download_that_is_not_blender_orgs_file_is_not_unzipped(self):
        real = engine.REPO, engine.free_disk
        engine.REPO, engine.free_disk = Path(self.folder.name), lambda path: 100 << 30
        try:
            source, open_url, _ = self.blender_zip({"blender-test/blender.exe": "exe"})
            with self.assertRaisesRegex(engine.StepFailed, "isn't the file blender.org lists"):
                engine.install_blender(open_url=open_url, source=dict(source, sha256="0" * 64), addon=False)
            home = Path(self.folder.name) / "tools" / "blender"
            self.assertEqual(list(home.iterdir()), [])
            self.assertIsNone(engine._read_json(engine.SETTINGS, {}).get("blender"))
            engine.free_disk = lambda path: 1 << 30
            with self.assertRaisesRegex(engine.StepFailed, "needs about 3 GB free"):
                engine.install_blender(open_url=open_url, source=source, addon=False)
        finally:
            engine.REPO, engine.free_disk = real

    def test_linux_gets_blender_orgs_tarball_unpacked_the_same_way(self):
        import hashlib
        import io
        import tarfile
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:xz") as packed:
            for name, text in (("blender-test/blender", b"elf"), ("blender-test/5.2/scripts/x.py", b"x")):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(text), 0o755
                packed.addfile(info, io.BytesIO(text))
        raw = data.getvalue()
        source = {"version": "5.2.2", "folder": "blender-test", "exe": "blender", "bytes": len(raw),
                  "url": "https://example.invalid/blender-test.tar.xz", "sha256": hashlib.sha256(raw).hexdigest()}
        real = engine.REPO, engine.blender_version, engine.free_disk
        engine.REPO, engine.blender_version, engine.free_disk = Path(self.folder.name), lambda path: "5.2.2", lambda path: 100 << 30
        try:
            engine.install_blender(open_url=lambda request, timeout=None: io.BytesIO(raw), source=source, addon=False)
            home = Path(self.folder.name) / "tools" / "blender"
            self.assertTrue((home / "blender-test" / "blender").is_file())
            self.assertTrue((home / "blender-test" / "portable").is_dir())
            self.assertEqual(sorted(p.name for p in home.iterdir()), ["blender-test"])
            self.assertEqual(Path(engine._read_json(engine.SETTINGS, {})["blender"]), home / "blender-test" / "blender")
            self.assertEqual(engine.portable_blenders(), [home / "blender-test" / "blender"])
        finally:
            engine.REPO, engine.blender_version, engine.free_disk = real

    def test_a_blender_whose_folder_names_no_version_is_asked(self):
        # macOS: /Applications/Blender.app/Contents/MacOS/Blender. By its folder alone it was never found
        real = engine.blender_version, engine.installed_blenders, engine.portable_blenders, engine.vrm_addon_version
        mac = Path(self.folder.name) / "Blender.app" / "Contents" / "MacOS" / "Blender"
        mac.parent.mkdir(parents=True)
        mac.write_text("")
        asked = []
        try:
            engine.blender_version = lambda path: asked.append(str(path)) or "5.2.2"
            engine.installed_blenders, engine.portable_blenders = lambda: [mac], lambda: []
            engine.vrm_addon_version = lambda path, version: "4.7.2"
            self.assertEqual(engine._folder_version(mac), (5, 2, 2))
            self.assertEqual(engine._folder_version(Path("/opt/blender-4.5.3-linux-x64/blender")), (4, 5, 3))
            self.assertEqual(len(asked), 1)  # a folder that names its version isn't asked
            listed = [b for b in engine.known_blenders() if Path(b["path"]) == mac]
            self.assertEqual([(b["version"], b["series"], b["tested"]) for b in listed], [("5.2.2", "5.2", True)])
        finally:
            engine.blender_version, engine.installed_blenders, engine.portable_blenders, engine.vrm_addon_version = real

    def test_on_a_mac_browse_asks_the_systems_own_dialog(self):
        # Tk off the first thread takes Python down on macOS, and the server answers on another
        real = server.sys, server.subprocess.run
        ran = []

        def run(command, **kwargs):
            ran.append(command)
            return SimpleNamespace(stdout="/Users/t/model.fbx" + chr(10), returncode=0)
        try:
            server.sys, server.subprocess.run = SimpleNamespace(platform="darwin"), run
            self.assertEqual(server._browse("model"), "/Users/t/model.fbx")
            self.assertEqual(ran[0][:2], ["osascript", "-e"])
            self.assertIn('choose file with prompt "Choose the model"', ran[0][2])
            server.subprocess.run = lambda command, **kwargs: SimpleNamespace(stdout="", returncode=1)  # Cancel
            self.assertIsNone(server._browse("rig"))
        finally:
            server.sys, server.subprocess.run = real

    def test_which_computers_the_app_fetches_blender_for(self):
        self.assertIs(engine.blender_build("win32", "AMD64"), engine.BLENDER_ZIP)
        self.assertIs(engine.blender_build("linux", "x86_64"), engine.BLENDER_TAR)
        self.assertIsNone(engine.blender_build("darwin", "arm64"))
        self.assertIsNone(engine.blender_build("linux", "aarch64"))  # blender.org's Linux build is x64
        for build in engine.BLENDER_BUILDS.values():
            self.assertTrue(build["url"].startswith("https://download.blender.org/release/") and build["folder"] in build["url"])
            self.assertEqual(len(build["sha256"]), 64)

    def test_the_addon_is_looked_for_where_each_system_keeps_blenders_settings(self):
        home = Path("/home/tester")
        self.assertEqual(engine.blender_settings("5.2", "linux", {}, home), home / ".config" / "blender" / "5.2")
        self.assertEqual(engine.blender_settings("5.2", "linux", {"XDG_CONFIG_HOME": "/cfg"}, home), Path("/cfg") / "blender" / "5.2")
        self.assertEqual(engine.blender_settings("5.2", "darwin", {}, home), home / "Library" / "Application Support" / "Blender" / "5.2")
        self.assertEqual(engine.blender_settings("4.5", "win32", {"APPDATA": "C:/Users/t/AppData/Roaming"}, home),
                         Path("C:/Users/t/AppData/Roaming") / "Blender Foundation" / "Blender" / "4.5")
        # ... and found there: an add-on installed on Linux no longer reads as missing
        settings = Path(self.folder.name) / ".config" / "blender" / "5.2"
        (settings / "extensions" / "blender_org" / "vrm").mkdir(parents=True)
        (settings / "extensions" / "blender_org" / "vrm" / "blender_manifest.toml").write_text('id = "vrm"\nversion = "4.7.2"\n')
        real = engine.blender_settings
        try:
            engine.blender_settings = lambda series: real(series, "linux", {}, self.folder.name)
            self.assertEqual(engine.vrm_addon_version("/opt/blender-5.2/blender", "5.2.2"), "4.7.2")
        finally:
            engine.blender_settings = real

    def test_what_is_kept_with_the_app_is_said_as_fully_local(self):
        real = engine.vendor.status
        try:
            engine.vendor.status = lambda: {"views": True, "tracker": True}
            row = next(c for c in engine.setup()["checks"] if c["name"] == "Fully local")
            self.assertEqual((row["state"], "action" in row), ("ok", False))
            self.assertIn("no connection is needed", row["text"])
            engine.vendor.status = lambda: {"views": False, "tracker": False}
            row = next(c for c in engine.setup()["checks"] if c["name"] == "Fully local")
            self.assertEqual((row["state"], row["action"]), ("warn", "vendor"))
        finally:
            engine.vendor.status = real

    def test_only_the_apps_own_packages_are_installed(self):
        self.assertEqual(engine.package_command("pillow")[1:], ["-m", "pip", "install", "pillow"])
        with self.assertRaisesRegex(ValueError, "isn't a package the app installs"):
            engine.package_command("requests; rm -rf")
        with self.assertRaisesRegex(ValueError, "isn't a package the app installs"):
            server.api("install_package", {}, {"name": "numpy"})

    def test_the_addon_is_installed_by_blender_itself(self):
        self.assertEqual(engine.addon_command("b.exe"), ["b.exe", "--online-mode", "--command", "extension", "install", "--sync", "--enable", "vrm"])

    def test_a_certificate_refused_on_windows_is_asked_for_again_against_the_roots(self):
        # blender.org's good certificate was refused for an expired intermediate Windows had kept
        import ssl
        import urllib.error
        import urllib.request
        from app import trust

        def refused(why):
            def first(request, timeout=None):
                raise urllib.error.URLError(why)
            return first
        real, real_system, again = urllib.request.urlopen, trust.sys.platform, []
        urllib.request.urlopen = lambda request, timeout=None, context=None: again.append((request, timeout, context)) or "answer"
        try:
            trust.sys.platform = "win32"
            if hasattr(ssl, "enum_certificates"):
                self.assertEqual(trust.open_url("https://x", timeout=5, first=refused(ssl.SSLCertVerificationError("expired"))), "answer")
                self.assertEqual(again[0][:2], ("https://x", 5))
                self.assertEqual((again[0][2].verify_mode, again[0][2].check_hostname), (ssl.CERT_REQUIRED, True))
            # no connection, or a refusal on a system whose list is roots already, is no reason to ask again
            with self.assertRaises(urllib.error.URLError):
                trust.open_url("https://x", first=refused(OSError("no route")))
            trust.sys.platform = "linux"
            with self.assertRaises(urllib.error.URLError):
                trust.open_url("https://x", first=refused(ssl.SSLCertVerificationError("expired")))
            self.assertEqual(len(again), 1 if hasattr(ssl, "enum_certificates") else 0)
        finally:
            urllib.request.urlopen, trust.sys.platform = real, real_system

    def test_a_failed_install_says_what_the_command_said(self):
        with self.assertRaisesRegex(engine.StepFailed, "installing nothing failed \\(exit code 3\\)(.|\\n)*no such thing"):
            engine.install([sys.executable, "-c", "import sys; print('no such thing'); sys.exit(3)"], "installing nothing")
        events = []
        done = engine.install([sys.executable, "-c", "print('fine')"], "installing nothing", on_event=events.append)
        self.assertEqual((done["installed"], done["log"].strip(), [e["status"] for e in events]), ("installing nothing", "fine", ["running", "done"]))


class Updates(unittest.TestCase):
    """The Update button: the app's own code replaced from the published zip, the user's things left alone."""

    OLD, NEW = "a" * 40, "b" * 40

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = Path(self.folder.name)
        self.real = update.REPO, update.CONFIG, update.VERSION
        update.REPO, update.CONFIG, update.VERSION = self.app, self.app / "update.json", self.app / ".oc2vrm_version.json"
        self.real_number, update.NUMBER = update.NUMBER, self.app / "VERSION"
        update.NUMBER.write_text("1\n")
        for name, text in {"app/server.py": "old server", "app/engine.py": "engine", "app/gone.py": "dropped by the next version",
                           "OC2VRM.bat": "old launcher", "out/Mine/character.json": "my character", "tools/blender/blender.exe": "blender",
                           "music.mp3": "my tune"}.items():
            (self.app / name).parent.mkdir(parents=True, exist_ok=True)
            (self.app / name).write_text(text)
        update.CONFIG.write_text(json.dumps({"repo": "Owner/Name", "ref": "testers", "token": "key"}))
        update.VERSION.write_text(json.dumps({"commit": self.OLD, "files": ["app/server.py", "app/engine.py", "app/gone.py", "OC2VRM.bat"]}))
        self.asked = []

    def tearDown(self):
        update.REPO, update.CONFIG, update.VERSION = self.real
        update.NUMBER = self.real_number
        self.folder.cleanup()

    def github(self, files=None, latest=None, fail=None):
        """A stand-in for api.github.com: the newest commit of the branch, and that commit's zip."""
        import io
        import urllib.error
        import zipfile
        files = files if files is not None else {"app/server.py": "new server", "app/engine.py": "engine", "app/new.py": "added",
                                                 "OC2VRM.bat": "new launcher", "out/Mine/character.json": "not theirs to write",
                                                 "update.json": "nor this", "README.md": "read me"}
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for name, text in files.items():
                archive.writestr(f"Owner-Name-bbbbbbb/{name}", text)

        def open_url(request, timeout=None):
            self.asked.append((request.full_url, request.get_header("Authorization")))
            if fail:
                raise urllib.error.HTTPError(request.full_url, fail, "no", {}, None)
            if "/contents/VERSION" in request.full_url:
                return io.BytesIO(b"2\n")
            return io.BytesIO((latest or self.NEW).encode() if "/commits/" in request.full_url else data.getvalue())
        return open_url

    def test_a_newer_version_is_seen_and_asked_for_with_the_key(self):
        found = update.check(self.github())
        self.assertEqual((found["configured"], found["current"], found["latest"], found["available"], found["how"]),
                         (True, "aaaaaaa", "bbbbbbb", True, "zip"))
        self.assertEqual(self.asked[0], ("https://api.github.com/repos/Owner/Name/commits/testers", "Bearer key"))
        # the version people say: this copy's own number, and the published one's
        self.assertEqual((found["version"], found["latest_version"]), ("v1", "v2"))
        self.assertFalse(update.check(self.github(latest=self.OLD))["available"])
        self.assertIn("the key in update.json was refused", update.check(self.github(fail=401))["error"])
        update.CONFIG.unlink()
        asked = len(self.asked)
        self.assertEqual((update.check(self.github())["configured"], len(self.asked)), (False, asked))  # no file, nothing asked

    def test_a_checkout_ahead_of_the_release_is_offered_nothing(self):
        # the owner works on past a release: the published commit is in their history already
        real = update._holds
        update._holds = lambda commit: commit == self.NEW
        try:
            found = update.check(self.github())
        finally:
            update._holds = real
        self.assertEqual((found["latest"], found["available"]), ("bbbbbbb", False))
        self.assertFalse(update._holds(self.NEW))  # a zip holds no history

    def test_the_code_is_replaced_and_the_users_things_are_left_alone(self):
        events = []
        done = update.apply(on_event=events.append, open_url=self.github())
        self.assertEqual((done["updated"], done["was"], done["now"], done["restart"]), (True, "aaaaaaa", "bbbbbbb", True))
        read = lambda name: (self.app / name).read_text()  # noqa: E731
        self.assertEqual((read("app/server.py"), read("app/new.py"), read("README.md")), ("new server", "added", "read me"))
        # the user's own: a character, Blender, the key's file, a tune
        self.assertEqual((read("out/Mine/character.json"), read("tools/blender/blender.exe"), read("music.mp3")),
                         ("my character", "blender", "my tune"))
        self.assertEqual(json.loads(read("update.json"))["token"], "key")
        # a launcher isn't written over while it may be running: the new one waits beside it
        self.assertEqual((read("OC2VRM.bat"), read("OC2VRM.bat.new")), ("old launcher", "new launcher"))
        # what the last version brought and this one doesn't is gone, and kept in the backup with what was replaced
        self.assertFalse((self.app / "app" / "gone.py").exists())
        backup = Path(done["backup"])
        self.assertEqual(((backup / "app" / "server.py").read_text(), (backup / "app" / "gone.py").read_text()),
                         ("old server", "dropped by the next version"))
        self.assertEqual((done["files"], done["removed"]), (4, 1))  # engine.py was the same and wasn't touched
        self.assertEqual(update.current()["commit"], self.NEW)
        self.assertEqual(engine.app_version() if update.REPO == engine.REPO else update.current()["short"], "bbbbbbb")
        # again: nothing to do
        self.assertEqual(update.apply(open_url=self.github())["updated"], False)

    def test_a_zip_that_is_not_the_app_changes_nothing(self):
        with self.assertRaisesRegex(update.UpdateFailed, "isn't this app"):
            update.apply(open_url=self.github(files={"README.md": "something else"}))
        self.assertEqual((self.app / "app" / "server.py").read_text(), "old server")
        self.assertEqual(update.current()["commit"], self.OLD)
        with self.assertRaisesRegex(update.UpdateFailed, "wasn't found with this key"):
            update.apply(open_url=self.github(fail=404))

    def test_a_zip_path_that_leaves_the_app_on_windows_is_not_written(self):
        # "/x" and "C:x" are no absolute paths to Windows, and REPO / "/x" is the drive's own folder
        for name in ("/x", "\\x", "C:x", "C:/x", "a/../b", "a\\..\\b", "//server/share/x", "..", ""):
            self.assertFalse(update._guarded(name), name)
        for name in ("app/server.py", "OC2VRM.bat", "docs/agent/app.md"):
            self.assertTrue(update._guarded(name), name)

    def test_the_users_own_things_are_kept_however_a_zip_spells_them(self):
        # Windows reads "Out", "TOOLS" and "out." as out and tools, and "Update.JSON" as update.json:
        # a zip that spelled them so wrote over the characters, Blender and the key an update never touches
        for name in ("Out/OC3/character.json", "TOOLS/blender/blender.exe", "out./x.json", "samples /a.glb", "Update.JSON",
                     ".OC2VRM_version.json", "Music.MP3", ".GIT/config", "update.json.", "dist/x.zip"):
            self.assertFalse(update._guarded(name), name)
        for name in ("output/notes.md", "app/update.json.py", "tools.md"):
            self.assertTrue(update._guarded(name), name)

    def test_a_run_in_a_copy_without_git_says_which_version_built_it(self):
        # a tester's run.json said "commit": null: git was the only one asked
        made_by = engine.provenance("blender", self.app / "out" / "Mine" / "character.json")
        self.assertEqual((made_by["commit"], made_by["dirty"], made_by["app"]), (self.OLD, False, "v1"))
        update.VERSION.unlink()
        made_by = engine.provenance("blender", self.app / "out" / "Mine" / "character.json")
        self.assertEqual((made_by["commit"], made_by["dirty"]), (None, None))

    def test_the_testers_zip_holds_the_committed_files_and_says_which_commit(self):
        import shutil
        import zipfile
        update.REPO, update.CONFIG, update.VERSION = self.real
        if not shutil.which("git") or not (update.REPO / ".git").exists():
            self.skipTest("not a git checkout")
        packed = update.pack(self.app / "dist")
        with zipfile.ZipFile(packed["zip"]) as archive:
            names = set(archive.namelist())
            version = json.loads(archive.read("OC2VRM/.oc2vrm_version.json"))
        self.assertTrue({"OC2VRM/OC2VRM.bat", "OC2VRM/app/server.py", "OC2VRM/requirements.txt"} <= names)
        self.assertEqual(version["commit"][:7], packed["commit"])
        self.assertIn("app/server.py", version["files"])
        self.assertFalse(any(n.startswith(("OC2VRM/out/", "OC2VRM/tools/", "OC2VRM/samples/")) for n in names))
        # the owner's song goes with it when there is one beside the app, though it is never committed
        self.assertEqual(packed["music"], next((n for n in ("music.mp3", "music.ogg", "music.wav") if (update.REPO / n).is_file()), ""))
        self.assertEqual(f"OC2VRM/{packed['music']}" in names, bool(packed["music"]))
        # an earlier commit packs as it was: work goes on after a release, and the zip must be the release
        import subprocess
        before = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=update.REPO, capture_output=True, text=True).stdout.strip()
        if before:
            earlier = update.pack(self.app / "dist", at="HEAD~1")
            with zipfile.ZipFile(earlier["zip"]) as archive:
                self.assertEqual(json.loads(archive.read("OC2VRM/.oc2vrm_version.json"))["commit"], before)
        with self.assertRaisesRegex(update.UpdateFailed, "no such commit"):
            update.pack(self.app / "dist", at="no-such-branch")


class ServerAnswers(unittest.TestCase):
    """A request that goes wrong is answered with why, and kept: the page once said only "Failed to fetch"."""

    def setUp(self):
        import threading
        self.folder = tempfile.TemporaryDirectory()
        self.real = server.ERROR_LOG, server.api
        server.ERROR_LOG = Path(self.folder.name) / "app_errors.log"
        self.served, self.port = server._listen(0)
        threading.Thread(target=self.served.serve_forever, daemon=True).start()

    def tearDown(self):
        self.served.shutdown()
        self.served.server_close()
        server.ERROR_LOG, server.api = self.real
        self.folder.cleanup()

    def get(self, route):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/{route}", timeout=20) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_an_unexpected_failure_is_said_and_written_down(self):
        def broken(route, query, body):
            raise RuntimeError("the disk went away")
        server.api = broken
        code, said = self.get("projects")
        self.assertEqual(code, 500)
        self.assertIn("RuntimeError: the disk went away", said["error"])
        log = server.ERROR_LOG.read_text(encoding="utf-8")
        self.assertIn("projects", log)
        self.assertIn("Traceback", log)

    def test_one_character_that_cannot_be_read_does_not_empty_the_list(self):
        real = server.Project
        try:
            class Broken:
                def __init__(self, path):
                    raise TypeError("a state file nobody expected")
            server.Project = Broken
            self.assertEqual(server._summary(Path(self.folder.name) / "character.json"), {})
            self.assertIn("a state file nobody expected", server.ERROR_LOG.read_text(encoding="utf-8"))
        finally:
            server.Project = real

    def test_a_job_that_fails_says_what_failed_in_its_first_line(self):
        # the page shows an error's first line: of a traceback that is "Traceback (most recent call last):",
        # which is all a build started with no Blender on the computer said
        import time

        def ended(work):
            jobs = server.Jobs()
            jobs.start("build", work)
            for _ in range(200):
                if not jobs.running:
                    return jobs.poll()
                time.sleep(0.01)
            self.fail("the job never ended")

        def no_blender(emit):
            raise FileNotFoundError("Blender 4.2 or later not found: put it on PATH")
        said = ended(no_blender)["error"]
        self.assertEqual(said.splitlines()[0], "FileNotFoundError: Blender 4.2 or later not found: put it on PATH")
        self.assertIn("Traceback", said)
        log = server.ERROR_LOG.read_text(encoding="utf-8")
        self.assertIn("job build", log)
        self.assertIn("no_blender", log)

        def refused(emit):
            raise engine.StepFailed("build: failed with exit code 1\nthe log's last lines")
        self.assertEqual(ended(refused)["error"], "build: failed with exit code 1\nthe log's last lines")
        self.assertNotIn("refused", server.ERROR_LOG.read_text(encoding="utf-8"))
        self.assertEqual(ended(lambda emit: {"done": 1})["error"], None)

    def test_a_file_that_is_not_utf8_reads_as_missing_and_one_with_a_byte_order_mark_reads(self):
        # one character.json saved by an editor in Windows' own code page took the whole Characters page down
        folder = Path(self.folder.name)
        odd = folder / "odd.json"
        odd.write_bytes('{"name": "Jos\xe9", "source": "m.glb", "options": {}}'.encode("cp1252"))
        self.assertEqual(engine._read_json(odd, "missing"), "missing")
        self.assertEqual(server._name(odd), "odd")
        listed = folder / "listed.json"
        listed.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(server._name(listed), "listed")
        # as Notepad and PowerShell write UTF-8: the character was off the list, and opening it failed
        marked = folder / "marked.json"
        manifest.save(manifest.new("model.glb", "Marked"), marked)
        marked.write_bytes(b"\xef\xbb\xbf" + marked.read_bytes())
        self.assertEqual(engine._read_json(marked, None)["name"], "Marked")
        self.assertEqual(server._name(marked), "Marked")
        self.assertEqual(manifest.load(marked)["name"], "Marked")


class Server(unittest.TestCase):
    def test_serves_only_model_and_output_files_inside_the_repository(self):
        inside = REPO / "viewer" / "index.html"
        self.assertEqual(server._served(inside), "viewer/index.html")
        for path in (REPO / "out" / ".." / "README.md", REPO / "scripts" / "make_vrm.py", REPO / ".git" / "config",
                     REPO.parent / "outside.json", REPO / "viewer" / "missing.html"):
            self.assertIsNone(server._served(path), path)

    def test_model_route_serves_only_listed_characters(self):
        class Probe(server.Handler):
            def __init__(self):
                self.errors = []

            def send_error(self, code, *args):
                self.errors.append(code)

            def _file(self, path, root=None):
                self.errors.append(("file", path))
        probe = Probe()
        probe._model({"manifest": [str(REPO / "out" / "nobody" / "character.json")], "key": ["source"]})
        self.assertEqual(probe.errors, [403])

    def test_shows_only_files_the_build_made(self):
        for path in (REPO / "README.md", REPO / "out" / ".." / "scripts" / "make_vrm.py", REPO.parent):
            with self.assertRaisesRegex(ValueError, "only files the build made"):
                server._reveal(path)

    def test_render_folders_name_their_run_and_pose(self):
        match = lambda name: server.RENDER_FOLDER.match(name).groupdict()  # noqa: E731
        self.assertEqual(match("20260102-100000_09_tpose_t-pose"), {"run": "20260102-100000", "blend": "09_tpose", "pose": "t-pose"})
        self.assertEqual(match("09_tpose_raised"), {"run": None, "blend": "09_tpose", "pose": "raised"})

    def test_refuses_a_name_with_a_path(self):
        with self.assertRaisesRegex(ValueError, "path"):
            server.api("init", {}, {"model": "m.glb", "name": "../evil"})
        for name in ("..", ".", " .. "):  # ".." is its own .name: the character landed in the app's folder
            with self.assertRaisesRegex(ValueError, "path"):
                server.api("init", {}, {"model": "m.glb", "name": name})

    def test_refuses_a_name_no_folder_can_have(self):
        # Windows' own refusal of the folder came back as the app's crash: "[WinError 123] ... syntax is incorrect"
        for name, odd in (("Yuki?", r"\?"), ("Miku: winter", ":"), ('the "best" one', '"'), ("a|b*", r"\* \|"), ("two\tparts", "a tab")):
            with self.assertRaisesRegex(ValueError, f"can't have {odd}.* in it: it is also the name of the character's folder"):
                server.api("init", {}, {"model": "m.glb", "name": name})
        # with the settings file put somewhere by hand (the command line), the name is no folder's
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "m.glb"
            model.write_bytes(b"glTF")
            made = cli.cmd_init(SimpleNamespace(model=model, name="Yuki?", author="Unknown", manifest=Path(folder) / "c.json",
                                                face=None, reference=None, rig=None))
            self.assertEqual(manifest.load(made["manifest"])["name"], "Yuki?")

    def test_a_model_that_cant_be_read_is_refused_when_the_character_is_made(self):
        # a tester's good FBX failed the survey as an unsupported type: the path, pasted from
        # Explorer's "Copy as path", kept its quotes
        self.assertEqual(server._typed(' "C:\\art\\my oc.fbx" '), "C:\\art\\my oc.fbx")
        with tempfile.TemporaryDirectory() as folder:
            packed = Path(folder) / "tripo.zip"
            packed.write_bytes(b"PK")
            for model, said in ((packed, "unpack the .zip"), (Path(folder), "that is a folder"),
                                (Path(folder) / "gone.fbx", "no model file"), ("", "name the model file first")):
                with self.assertRaisesRegex(ValueError, said):
                    server.api("init", {}, {"model": f'"{model}"' if model else "", "name": "Refused"})
        self.assertFalse((REPO / "out" / "Refused").exists())

    def test_a_get_only_reads(self):
        # another site's <img src=".../api/prune?manifest=..."> reached every call: no header is asked of a GET
        class Probe(server.Handler):
            def __init__(self, path):
                self.path, self.headers, self.said = path, {"Host": "127.0.0.1:1"}, []
                self.server = SimpleNamespace(server_port=1)

            def _json(self, data, code=200):
                self.said.append(code)

            def _api(self, route, query, body):
                self.said.append(route)
        for route in ("run", "prune", "approve", "cancel", "vendor", "sheet", "face_guess", "send", "set", "bundle", "report", "report_send", "report_address"):
            probe = Probe(f"/api/{route}?manifest=x")
            probe.do_GET()
            self.assertEqual(probe.said, [405], route)
        for route in server.READ_ONLY:
            probe = Probe(f"/api/{route}")
            probe.do_GET()
            self.assertEqual(probe.said, [route])

    def asked(self, path, **headers):
        """(status, headers) of a GET to a server started for it, with the headers a browser would add."""
        import http.client
        import threading
        served, port = server._listen(0)
        threading.Thread(target=served.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            response.read()
            return response.status, {name.lower(): value for name, value in response.getheaders()}
        finally:
            connection.close()
            served.shutdown()
            served.server_close()

    def test_a_page_of_another_site_is_refused(self):
        # any site could send the browser to /viewer/index.html?model=<its own file>, whose bone names the
        # viewer wrote into the page as HTML, in a page with the app's rights (every character, every POST)
        viewer = "/viewer/index.html?model=https://elsewhere.example/a.vrm"
        for site in ("cross-site", "same-site"):  # same-site: another program's page on this computer
            for path in (viewer, "/api/projects", "/gui/app.css", "/?character=x"):
                self.assertEqual(self.asked(path, **{"Sec-Fetch-Site": site, "Sec-Fetch-Mode": "navigate"})[0], 403, (site, path))
        # the app's own page, the address typed or opened by the app, and a program that is no browser
        for site in ("same-origin", "none", None):
            self.assertEqual(self.asked("/viewer/index.html", **({"Sec-Fetch-Site": site} if site else {}))[0], 200, site)
        # a link to the app on another site's page still opens its first page, and nothing else
        self.assertEqual(self.asked("/", **{"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"})[0], 200)
        self.assertEqual(self.asked("/", **{"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "iframe"})[0], 403)

    def test_answers_say_how_a_browser_may_use_them(self):
        for path in ("/", "/gui/app.css", "/api/setup", "/viewer/index.html"):
            status, headers = self.asked(path)
            self.assertEqual(status, 200, path)
            self.assertEqual(headers.get("x-content-type-options"), "nosniff", path)          # a .log or .json is never read as a page
            self.assertEqual(headers.get("cross-origin-resource-policy"), "same-origin", path)  # nor a picture shown on another site
            self.assertEqual(headers.get("x-frame-options"), "SAMEORIGIN", path)               # nor the app framed by one
        self.assertEqual(self.asked("/gui/app.css")[1]["content-type"], "text/css")  # whatever Windows' registry says of .css

    def test_the_viewer_writes_no_name_of_a_model_as_html_and_opens_only_this_servers_models(self):
        page = (REPO / "viewer" / "index.html").read_text(encoding="utf-8")
        # Edge 155 makes elements of markup inside a <select>'s options: a bone named <b>x or <svg> became
        # one (out/_task5/viewer_probe.py); no script ran, only because three.js strips " ", ".", ":" and "/"
        # from a node's name
        self.assertNotRegex(page, r"innerHTML\s*\+?=[^;]*\$\{name\}")
        self.assertIn("new Option(name, name)", page)
        self.assertRegex(page, r"new URL\(model, location\.href\)\.origin === location\.origin")

    def test_a_running_step_shows_the_end_of_its_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "01_build.log"
            log.write_bytes(b"x" * 5000 + b"\nskinning the arms\n")
            jobs = server.Jobs()
            jobs.running, jobs.events = "build", [{"step": "survey", "status": "done"}, {"step": "build", "status": "running", "log": str(log)}]
            tail = jobs.poll()["tail"]
            self.assertTrue(tail.endswith("skinning the arms\n") and len(tail) <= server.LOG_TAIL)
            jobs.events.append({"step": "hair", "status": "running", "log": str(Path(folder) / "02_hair.log")})
            self.assertIsNone(jobs.poll()["tail"])  # not written yet
            jobs.running = None
            self.assertIsNone(jobs.poll()["tail"])

    def test_sheets_are_listed_by_run(self):
        (REPO / "out").mkdir(exist_ok=True)  # served files live under out/, which a fresh clone lacks
        with tempfile.TemporaryDirectory(dir=REPO / "out") as folder:
            renders = Path(folder) / "renders"
            for run in ("20260101-100000", "20260102-100000"):
                (renders / f"sheets_{run}").mkdir(parents=True)
                (renders / f"sheets_{run}" / "sheet_face.png").write_bytes(b"png")
            (renders / "sheets_20260102-100000" / "sheet.json").write_text(json.dumps({"missing": ["blink"]}))
            (renders / "sheets_empty").mkdir()
            found = {s["run"]: s for s in server._sheets(renders)}
            self.assertEqual(sorted(found), ["20260101-100000", "20260102-100000"])
            self.assertEqual(found["20260102-100000"]["report"], {"missing": ["blink"]})
            self.assertTrue(found["20260101-100000"]["urls"][0].endswith("/renders/sheets_20260101-100000/sheet_face.png"))

    def test_a_second_start_finds_the_first(self):
        import threading
        first, port = server._listen(0)
        self.assertIsNotNone(first)
        threading.Thread(target=first.serve_forever, daemon=True).start()
        try:
            self.assertTrue(server._ours(port))
            self.assertEqual(server._listen(port), (None, port))
        finally:
            first.shutdown()
            first.server_close()
        self.assertFalse(server._ours(port))

    def test_only_character_manifests_are_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "samples" / "manifests").mkdir(parents=True)
            manifest.save(manifest.new("model.glb", "Real"), root / "samples" / "manifests" / "Real.json")
            # face points kept beside the manifests showed as a character called Name_face
            (root / "samples" / "manifests" / "Real_face.json").write_text(json.dumps({"image": "p.png", "size": [8, 8], "mouth": {}}))
            (root / "samples" / "manifests" / "broken.json").write_text("{")
            real, engine.REPO = engine.REPO, root
            try:
                self.assertEqual([p.name for p in server._characters()], ["Real.json"])
            finally:
                engine.REPO = real


class Paint(unittest.TestCase):
    """Texture paint kept as overlays and laid over the source for a build (app/paint.py)."""

    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            # Blender's own Python, which runs this file too with the suite, has no Pillow; the app's has
            self.skipTest("Pillow is not in this Python: run with python -m unittest tests.test_app")
        self.Image = Image
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        # a model with two pictures, the first a normal map no material paints on, and its geometry first in the file
        self.geometry = bytes(range(40))
        views, binary = [], bytearray()
        for data in (self.geometry, self.png((8, 8), (10, 20, 30, 255)), self.png((4, 4), (200, 100, 50, 255))):
            binary += b"\0" * (-len(binary) % 4)
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
            binary += data
        gltf = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0], "extras": {"vtuber_rig": "rig.fbx"}}], "nodes": [{"mesh": 0}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
                "images": [{"name": "normal", "mimeType": "image/png", "bufferView": 1}, {"name": "base colour", "mimeType": "image/png", "bufferView": 2}],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [{"name": "skin", "normalTexture": {"index": 0}, "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}}]}
        self.source = self.root / "model.glb"
        paint.write_glb(self.source, gltf, binary)
        self.path = self.root / "character.json"
        data = manifest.new("model.glb", "Test")
        data["out_dir"] = "out"
        manifest.save(data, self.path)

    def tearDown(self):
        if hasattr(self, "folder"):
            self.folder.cleanup()

    def png(self, size, colour):
        import io
        out = io.BytesIO()
        self.Image.new("RGBA", size, colour).save(out, "PNG")
        return out.getvalue()

    def url(self, raw):
        import base64
        return "data:image/png;base64," + base64.b64encode(raw).decode()

    def pixels(self, glb, image):
        import io
        gltf, binary = paint.read_glb(glb)
        view = gltf["bufferViews"][gltf["images"][image]["bufferView"]]
        return self.Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])).convert("RGBA")

    def test_only_base_colour_pictures_take_paint(self):
        self.assertEqual(cli.paint_state(self.path)["pictures"],
                         [{"index": 1, "name": "base colour", "overlay": "base_colour.png", "materials": ["skin"], "painted": False, "draft": False}])
        with self.assertRaisesRegex(ValueError, "not one of this model's pictures"):
            cli.save_paint(self.path, {"normal.png": self.url(self.png((8, 8), (255, 0, 0, 255)))})
        with self.assertRaisesRegex(ValueError, "not one of this model's pictures"):
            cli.save_paint(self.path, {"../evil.png": self.url(self.png((4, 4), (255, 0, 0, 255)))})

    def test_a_build_reads_the_painted_copy_and_nothing_else_changes(self):
        project = Project(self.path, blender="blender")
        self.assertEqual(project.model(), self.source.resolve())
        project.state["steps"] = {"survey": {"status": "done"}, "build": {"status": "done"}, "hair": {"status": "done"}}
        project._save_state()
        # half-opaque red over the lower half
        over = self.Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        over.paste((255, 0, 0, 128), (0, 2, 4, 4))
        import io
        raw = io.BytesIO()
        over.save(raw, "PNG")
        self.assertEqual(cli.save_paint(self.path, {"base_colour.png": self.url(raw.getvalue())})["painted"], ["base_colour.png"])
        project = Project(self.path, blender="blender")
        self.assertEqual(manifest.path_of(project.data, "paint"), (self.root / "out" / "paint" / "overlays").resolve())
        self.assertEqual({n: e["status"] for n, e in project.state["steps"].items()}, {"survey": "done", "build": "stale", "hair": "stale"})
        painted = project.model()
        self.assertEqual(painted, self.root.resolve() / "out" / "paint" / "Test_painted.glb")
        self.assertEqual(project._arguments("build", self.root / "out" / "runs" / "r" / "01_build", None)[0][1], painted)
        self.assertEqual(self.pixels(painted, 1).getpixel((0, 0)), (200, 100, 50, 255))
        self.assertEqual(self.pixels(painted, 1).getpixel((0, 3)), (228, 50, 25, 255))
        self.assertEqual(self.pixels(painted, 0).getpixel((0, 0)), (10, 20, 30, 255))
        before, after = paint.read_glb(self.source), paint.read_glb(painted)
        for key in ("scenes", "nodes", "textures", "materials"):
            self.assertEqual(before[0][key], after[0][key])
        view = after[0]["bufferViews"][0]
        self.assertEqual(after[1][view["byteOffset"]:view["byteOffset"] + view["byteLength"]], self.geometry)
        self.assertEqual(manifest.rig_of(painted), None)  # read without error: the scene's extras came along
        self.assertEqual(manifest._scene_extras(painted), {"vtuber_rig": "rig.fbx"})

    def test_the_copy_is_made_again_when_the_paint_or_the_model_changes(self):
        cli.save_paint(self.path, {"base_colour.png": self.url(self.png((4, 4), (0, 255, 0, 255)))})
        project = Project(self.path, blender="blender")
        painted = project.model()
        made = painted.stat().st_mtime_ns
        self.assertEqual(project.model().stat().st_mtime_ns, made)  # nothing changed: not made again
        cli.save_paint(self.path, {"base_colour.png": self.url(self.png((4, 4), (0, 0, 255, 255)))})
        self.assertEqual(self.pixels(Project(self.path, blender="blender").model(), 1).getpixel((1, 1)), (0, 0, 255, 255))
        # an overlay of another size than the picture (a face patch remade larger) is stretched over it
        cli.save_paint(self.path, {"base_colour.png": self.url(self.png((2, 2), (9, 9, 9, 255)))})
        self.assertEqual(self.pixels(Project(self.path, blender="blender").model(), 1).size, (4, 4))

    def test_cleared_paint_gives_the_source_back(self):
        cli.save_paint(self.path, {"base_colour.png": self.url(self.png((4, 4), (0, 255, 0, 255)))})
        cli.cmd_paint(SimpleNamespace(manifest=self.path, clear=True, blender="blender"))
        project = Project(self.path, blender="blender")
        self.assertIsNone(project.data["paint"])
        self.assertEqual(project.model(), self.source.resolve())

    def test_a_remade_face_patch_keeps_the_frame_and_starts_from_the_unpatched_model(self):
        folder = self.root / "out" / "face"
        self.assertIsNone(cli._framing(folder))
        (folder / "views").mkdir(parents=True)
        # the camera of the patch before, 0.8446 up a model 0.9995 tall: the face points are pixels in that frame
        (folder / "views" / "views.json").write_text(json.dumps({"height": 0.9995, "views": [
            {"name": "head_000", "matrix_world": [[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, 0.8446], [0, 0, 0, 1]]}]}))
        self.assertEqual(cli._framing(folder), 0.845)
        (folder / "head_at.json").write_text(json.dumps({"head_at": 0.808}))
        self.assertEqual(cli._framing(folder), 0.808)
        # a duplicate's source is its original's patched copy, outside its own folder: still not patched again
        self.assertFalse(cli._patched(self.source))
        gltf, binary = paint.read_glb(self.source)
        gltf["materials"].append({"name": "face_patch"})
        paint.write_glb(self.root / "patched.glb", gltf, binary)
        self.assertTrue(cli._patched(self.root / "patched.glb"))
        self.assertFalse(cli._patched(self.root / "model.fbx"))

    def test_the_face_view_is_framed_on_the_body_under_horns_or_a_tail(self):
        # eyes at 1.37 of a body 1.6 tall: by the body's share the view sat 23 cm too high on a model
        # 1.84 tall with its horns, the face at the picture's lower edge
        plain = {"size": [0.6, 0.4, 1.6], "eyes": [0, -0.12, 1.368], "joints": {"head": [0, -0.09, 1.347]}}
        self.assertEqual(cli._frame(plain, 1.6), (0.835, 1.0))
        horned = dict(plain, size=[0.6, 0.4, 1.84])
        at, body = cli._frame(horned, 1.6)
        self.assertEqual((at, body), (0.726, 0.87))
        self.assertAlmostEqual(at * 1.84, 0.835 * 1.6, delta=0.002)  # the same place on the body
        # without modelled eyes, from the head joint; with no survey, the script's own proportion
        self.assertEqual(cli._frame({"size": [0.6, 0.4, 1.84], "joints": {"head": [0, 0, 1.347]}}, 1.6)[0], 0.767)
        self.assertEqual(cli._frame(None, 1.6), (None, 1.0))

    def test_a_chibi_head_gets_a_view_its_size(self):
        # the chibi girl, 2.5 heads: an adult-sized view cut her eyes at its edges and left out her mouth
        chibi = {"size": [0.9, 0.5, 1.6], "eyes": [0, -0.32, 1.178], "head": {"height": 0.58}}
        self.assertEqual(cli._frame(chibi, 1.6), (0.716, 2.522))
        grown = dict(chibi, head={"height": 0.25})
        self.assertEqual(cli._frame(grown, 1.6)[1], 1.0)

    def test_the_face_patch_has_only_the_views_drawn_that_something_reads(self):
        # render_views.py draws 34 views with their depths, for re-texturing by hand. Drawn so for every new
        # character, 25 of them and 33 depths were read by nothing: half of the face patch's seconds
        asked = []

        def ran(blender, script, args, log=None, **more):
            asked.append((script, [str(a) for a in args]))
            if script.endswith("render_views.py"):
                Path(args[1]).mkdir(parents=True, exist_ok=True)
                (Path(args[1]) / "color_head_000.png").write_bytes(self.png((8, 8), (200, 150, 120, 255)))
            else:
                Path(args[4]).write_bytes(b"glTF")
            return 0, ""
        real, engine.run_script = engine.run_script, ran
        try:
            cli.cmd_face_patch(SimpleNamespace(manifest=self.path, blender="blender"))
        finally:
            engine.run_script = real
        (drew, views), (made, patch) = asked
        self.assertEqual((drew, made), ("retexture/render_views.py", "retexture/face_patch.py"))
        only = views[views.index("--only") + 1].split(",")
        # the head from the front, which the patch is made of, and its depth alone
        self.assertEqual((patch[2], only[0], views[views.index("--depth-of") + 1]), ("head_000", "head_000", "head_000"))
        # each part from the front and from behind: what a handoff bundle and a tester's report take
        self.assertEqual(sorted(only), sorted(cli.PATCH_VIEWS))
        self.assertEqual({name.rsplit("_", 1)[0] for name in only}, {"head", "torso", "hips", "legs", "face"})
        self.assertTrue(all(name.endswith(("_000", "_180")) for name in only))
        # ... by the names the script gives its views, which stops on one it doesn't know
        script = (REPO / "scripts" / "retexture" / "render_views.py").read_text(encoding="utf-8")
        for tile in ("head", "torso", "hips", "legs"):
            self.assertIn(f'("{tile}", ', script)
        self.assertIn('"face_000"', script)
        self.assertIn("no such view", script)
        # the marking picture is the head view, twice the size
        with self.Image.open(self.root / "out" / "face" / "face_patch_head_000.png") as picture:
            self.assertEqual(picture.size, (2048, 2048))

    def test_unsaved_paint_is_kept_as_a_draft_until_it_is_saved(self):
        project = Project(self.path, blender="blender")
        project.state["steps"] = {"build": {"status": "done"}}
        project._save_state()
        cli.save_paint_draft(self.path, {"base_colour.png": self.url(self.png((4, 4), (1, 2, 3, 255)))})
        state = cli.paint_state(self.path)["pictures"][0]
        self.assertEqual((state["painted"], state["draft"]), (False, True))
        # a draft is not the character's paint: the build reads the source, and nothing is stale
        project = Project(self.path, blender="blender")
        self.assertEqual(project.model(), self.source.resolve())
        self.assertEqual(project.state["steps"]["build"]["status"], "done")
        self.assertIsNone(project.data["paint"])
        cli.save_paint(self.path, {"base_colour.png": self.url(self.png((4, 4), (1, 2, 3, 255)))})
        state = cli.paint_state(self.path)["pictures"][0]
        self.assertEqual((state["painted"], state["draft"]), (True, False))
        self.assertFalse((self.root / "out" / "paint" / "draft").exists())

    def test_an_older_face_patch_is_named(self):
        self.assertFalse(cli.paint_state(self.path)["old_patch"])  # no patch at all
        gltf, binary = paint.read_glb(self.source)
        gltf["materials"].append({"name": "face_patch"})
        paint.write_glb(self.source, gltf, binary)
        self.assertTrue(cli.paint_state(self.path)["old_patch"])
        (self.root / "patch.json").write_text(json.dumps({"version": cli.PATCH_VERSION}))
        self.assertFalse(cli.paint_state(self.path)["old_patch"])

    def test_paint_for_a_picture_the_model_lost_is_left_out(self):
        folder = self.root / "out" / "paint" / "overlays"
        folder.mkdir(parents=True)
        (folder / "face_patch_head_000.png").write_bytes(self.png((4, 4), (1, 2, 3, 255)))
        data = manifest.load(self.path)
        data["paint"] = "out/paint/overlays"
        manifest.save(data, self.path)
        self.assertEqual(Project(self.path, blender="blender").model(), self.source.resolve())
        self.assertEqual(cli.paint_state(self.path)["orphans"], ["face_patch_head_000.png"])


class Pictures(unittest.TestCase):
    """The pictures of an exported file cut to what the avatar uses, its normal map halved (app/slim.py)."""

    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            # Blender's own Python, which runs this file too with the suite, has no Pillow; the app's has
            self.skipTest("Pillow is not in this Python: run with python -m unittest tests.test_app")
        self.Image = Image
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        # every texel its own colour, so a cut a texel off shows
        self.face = Image.frombytes("RGB", (256, 256), bytes(v for y in range(256) for x in range(256) for v in (x, y, (x * 7 + y * 13) % 256)))

    def tearDown(self):
        if hasattr(self, "folder"):
            self.folder.cleanup()

    def png(self, picture):
        import io
        from PIL import PngImagePlugin
        said = PngImagePlugin.PngInfo()
        said.add(b"gAMA", struct.pack(">I", 45455))  # Blender writes each picture's colour space into its file
        out = io.BytesIO()
        picture.save(out, "PNG", pnginfo=said)
        return out.getvalue()

    def avatar(self, name="avatar.vrm", corners=(0.30, 0.35, 0.60, 0.70), face=None, normal=None, change=None):
        """A small exported avatar: a body whose picture and normal map its triangles use all over, and a
        face whose four corners read the middle of a 256 px picture. `change(gltf)` has the last word."""
        u0, v0, u1, v1 = corners
        floats = lambda *values: struct.pack(f"<{len(values)}f", *values)  # noqa: E731
        parts = [floats(*range(12)), floats(0, 0, 1, 0, 1, 1, 0, 1),              # the body's positions and UVs
                 floats(*range(12)), floats(u0, v0, u1, v0, u1, v1, u0, v1),      # the face's
                 self.png(self.Image.new("RGB", (64, 64), (90, 120, 150))),
                 normal or self.png(self.Image.new("RGB", (64, 64), (128, 128, 255))),
                 face or self.png(self.face)]
        binary, views = bytearray(), []
        for data in parts:
            binary += b"\0" * (-len(binary) % 4)
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
            binary += data
        vec = lambda view, kind: {"bufferView": view, "componentType": 5126, "count": 4, "type": kind}  # noqa: E731
        toon = lambda texture: {"VRMC_materials_mtoon": {"shadeMultiplyTexture": {"index": texture}}}  # noqa: E731
        gltf = {"asset": {"version": "2.0"}, "nodes": [{}] * 20, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
                "accessors": [vec(0, "VEC3"), vec(1, "VEC2"), vec(2, "VEC3"), {**vec(3, "VEC2"), "min": [u0, v0], "max": [u1, v1]}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0},
                                           {"attributes": {"POSITION": 2, "TEXCOORD_0": 3}, "material": 1}]}],
                "images": [{"name": "body", "bufferView": 4}, {"name": "body_normal", "bufferView": 5}, {"name": "face_patch", "bufferView": 6}],
                "samplers": [{"magFilter": 9729, "minFilter": 9985}],
                "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}, {"source": 2, "sampler": 0}],
                "materials": [{"name": "body", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}, "normalTexture": {"index": 1},
                               "extensions": toon(0)},
                              {"name": "face_patch", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}}, "extensions": toon(2)}],
                "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": {b: {"node": i} for i, b in enumerate(sorted(engine.REQUIRED_BONES))}}}}}
        if change:
            change(gltf)
        path = self.root / name
        paint.write_glb(path, gltf, binary)
        return path

    def picture(self, path, image):
        import io
        gltf, binary = paint.read_glb(path)
        view = gltf["bufferViews"][gltf["images"][image]["bufferView"]]
        return self.Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]))

    def numbers(self, path, accessor):
        gltf, binary = paint.read_glb(path)
        entry = gltf["accessors"][accessor]
        count = entry["count"] * {"VEC2": 2, "VEC3": 3}[entry["type"]]
        return struct.unpack_from(f"<{count}f", binary, gltf["bufferViews"][entry["bufferView"]]["byteOffset"])

    def test_a_picture_is_cut_to_what_its_triangles_use(self):
        path = self.avatar()
        whole, was = path.stat().st_size, {a: self.numbers(path, a) for a in range(4)}
        pictures = {i: self.picture(path, i).tobytes() for i in (0, 1)}
        report = slim.pictures(path)
        # corners on multiples of 16 texels, 16 clear of the triangles
        self.assertEqual(report["cut"], [{"picture": "face_patch", "from": [256, 256], "to": [128, 144]}])
        self.assertEqual(report["halved"], [])  # a normal map of 64 px is small already
        self.assertEqual(report["loaded_megabytes"], [0.4, 0.1])
        cut = self.picture(path, 2)
        self.assertEqual(cut.size, (128, 144))
        self.assertEqual(cut.tobytes(), self.face.crop((48, 64, 176, 208)).tobytes())  # its texels, as they were
        self.assertAlmostEqual(cut.info["gamma"], 0.45455)                              # and what its file said of them
        now = self.numbers(path, 3)
        for i in range(0, 8, 2):  # each corner reads the texel it read
            self.assertAlmostEqual(now[i] * 128 + 48, was[3][i] * 256, places=3)
            self.assertAlmostEqual(now[i + 1] * 144 + 64, was[3][i + 1] * 256, places=3)
        bounds = paint.read_glb(path)[0]["accessors"][3]
        self.assertEqual((bounds["min"], bounds["max"]), ([min(now[0::2]), min(now[1::2])], [max(now[0::2]), max(now[1::2])]))
        # the body's picture, used all over, and all the rest are untouched
        self.assertEqual({i: self.picture(path, i).tobytes() for i in (0, 1)}, pictures)
        self.assertEqual({a: self.numbers(path, a) for a in range(3)}, {a: was[a] for a in range(3)})
        self.assertLess(path.stat().st_size, whole)
        self.assertEqual(slim.pictures(path), {"left": "no picture is bigger than the avatar uses"})  # done once

    def test_what_isnt_plainly_mapped_by_its_uvs_is_left_alone(self):
        import io
        jpeg = io.BytesIO()
        self.face.save(jpeg, "JPEG")
        toon = lambda gltf: gltf["materials"][1]["extensions"]["VRMC_materials_mtoon"]  # noqa: E731
        cases = {
            "used all over": {"corners": (0.02, 0.02, 0.98, 0.98)},
            "it repeats": {"corners": (0.30, 0.35, 1.20, 0.70)},
            "a JPEG can't be cut texel for texel": {"face": jpeg.getvalue()},
            "its UVs scroll": {"change": lambda gltf: toon(gltf).update(uvAnimationScrollXSpeedFactor=0.1)},
            "a matcap isn't mapped by UVs": {"change": lambda gltf: toon(gltf).update(matcapTexture={"index": 2})},
            "an expression moves its UVs": {"change": lambda gltf: gltf["extensions"]["VRMC_vrm"].update(
                expressions={"custom": {"wink": {"textureTransformBinds": [{"material": 1, "offset": [0.1, 0.0]}]}}})},
            "a texture transform": {"change": lambda gltf: gltf["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"].update(
                extensions={"KHR_texture_transform": {"offset": [0.1, 0.0]}})},
            "the body reads its UVs too": {"change": lambda gltf: gltf["meshes"][0]["primitives"][0]["attributes"].update(TEXCOORD_0=3)},
        }
        for number, (why, made) in enumerate(cases.items()):
            with self.subTest(why):
                path = self.avatar(f"kept_{number}.vrm", **made)
                whole = path.read_bytes()
                self.assertEqual(slim.pictures(path), {"left": "no picture is bigger than the avatar uses"})
                self.assertEqual(path.read_bytes(), whole)

    def test_a_big_normal_map_is_halved_and_its_vectors_made_unit(self):
        from unittest import mock
        normal = self.Image.new("RGB", (64, 64), (128, 128, 255))
        normal.paste((255, 128, 128), (0, 0, 2, 1))  # the first four texels: two lean one way,
        normal.paste((128, 255, 128), (0, 1, 2, 2))  # two another
        path = self.avatar(normal=self.png(normal))
        with mock.patch.object(slim, "NORMAL_FULL", 64):
            report = slim.pictures(path)
        self.assertEqual(report["halved"], [{"picture": "body_normal", "from": [64, 64], "to": [32, 32]}])
        self.assertEqual(len(report["cut"]), 1)  # the face, in the same go
        small = self.picture(path, 1)
        self.assertEqual(small.size, (32, 32))
        self.assertAlmostEqual(small.info["gamma"], 0.45455)
        self.assertEqual(small.getpixel((5, 5)), (128, 128, 255))  # four of one direction stay that direction
        # their mean, (191.5, 191.5, 128), is 90.5 from the middle; a unit vector is 127.5 from it
        self.assertEqual(small.getpixel((0, 0)), (218, 218, 128))

    def test_a_picture_that_cant_be_read_leaves_the_file_as_it_was(self):
        path = self.avatar(face=self.png(self.face)[:400])  # the start of a PNG and no more
        whole = path.read_bytes()
        report = slim.pictures(path)
        self.assertEqual(list(report), ["left"])
        self.assertEqual(path.read_bytes(), whole)

    def test_without_pillow_the_pictures_stay_at_full_size(self):
        from unittest import mock
        path = self.avatar()
        whole = path.read_bytes()
        with mock.patch.dict(sys.modules, {"PIL": None}):
            self.assertIn("Pillow isn't installed", slim.pictures(path)["left"])
        self.assertEqual(path.read_bytes(), whole)

    def test_the_export_step_ends_with_it_and_a_vrm_taken_in_keeps_its_pictures(self):
        exported = self.avatar("as_blender_writes_it.vrm").read_bytes()
        real = engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS

        def blender(blender, blend_in, args, log, extra):
            Path(log).write_text("wrote it\n")
            Path(args[1]).write_bytes(exported)
            return 0
        engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS = blender, lambda path: 100 << 30, lambda: 16 << 30, self.root / "settings.json"
        try:
            for source, cut in (("model.glb", True), ("avatar.vrm", False)):
                home = self.root / source.replace(".", "_")
                home.mkdir()
                data = manifest.new(source, "Test")
                data["out_dir"] = "out"
                manifest.save(data, home / "character.json")
                project = Project(home / "character.json", blender="blender")
                blend = home / "12_mouth.blend"
                blend.write_bytes(b"BLENDER-v402")
                project._record("build", {"status": "done", "blend": str(blend)})
                entry = project._run_step("export", engine.NAMES.index("export"), home)
                self.assertEqual(entry["status"], "done")
                self.assertEqual(project.improving, not cut)
                self.assertEqual(self.picture(entry["vrm"], 2).size, (128, 144) if cut else (256, 256))
                self.assertEqual((entry.get("report") or {}).get("pictures", {}).get("cut"),
                                 [{"picture": "face_patch", "from": [256, 256], "to": [128, 144]}] if cut else None)
                self.assertEqual("pictures: " in Path(entry["log"]).read_text(), cut)  # said in the step's log
        finally:
            engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS = real


class Pages(unittest.TestCase):
    """The app's page opened in a browser without a window: every module loads and runs to its end.
    A page that throws while loading shows a half-drawn app and no error anywhere a test would see."""

    # Read into the served page after main.js: once the page's first drawing is done (app.started), it sends
    # what it drew. The browser's own --dump-dom took its picture when its fast-forward clock
    # (--virtual-time-budget) ran out, and on Windows that came before the page's code had started, every run
    # (2026-10-08): the browser exited with a page asking /api/setup, and its page and network processes
    # went on for seconds, asking for /api/welcome into the next page's count and holding the profile.
    REPORT = """<script type="module">
import { app } from "/gui/js/core.js";
// main.js has run, or failed to and left no app.started: either way the page is as its code left it
Promise.resolve(app.started).finally(() => setTimeout(() => fetch(`/test/page${location.search}`, { method: "POST", body: document.documentElement.outerHTML })));
</script>
"""

    def test_every_page_module_runs(self):
        import shutil
        import threading
        import time
        from urllib.parse import parse_qs, urlparse
        from app import sheet
        try:
            browser = sheet.find_browser()
        except sheet.NoBrowser as error:
            self.skipTest(str(error))
        served, port = server._listen(0)
        threading.Thread(target=served.serve_forever, daemon=True).start()
        # a computer with everything a build needs, whatever this one has: the app then opens on Characters
        real_setup = engine.setup
        ready = lambda given=None: {**real_setup(given), "ready": True, "welcomed": True, "checks": [c for c in real_setup(given)["checks"] if c["state"] != "bad"]}  # noqa: E731
        engine.setup = ready
        # (the page says a first start was seen: not written into this computer's settings)
        real_welcome, welcomes = engine.welcome, []
        engine.welcome = lambda done=True: welcomes.append(done) or {"welcomed": bool(done)}

        # the page with its report, and what each opening's page sent back, by its number
        real_get, real_post, drawn, came = server.Handler.do_GET, server.Handler.do_POST, {}, threading.Condition()
        page = vendor.local((server.GUI / "index.html").read_text(encoding="utf-8"))
        self.assertIn("</body>", page)
        page = page.replace("</body>", self.REPORT + "</body>").encode("utf-8")

        def get(handler):
            if urlparse(handler.path).path != "/":
                return real_get(handler)
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.send_header("Content-Length", str(len(page)))
            handler.end_headers()
            handler.wfile.write(page)

        def post(handler):
            url = urlparse(handler.path)
            if url.path != "/test/page":
                return real_post(handler)
            body = handler.rfile.read(int(handler.headers.get("Content-Length") or 0)).decode("utf-8")
            with came:
                drawn[parse_qs(url.query)["run"][0]] = body
                came.notify_all()
            handler.send_response(204)
            handler.end_headers()
        server.Handler.do_GET, server.Handler.do_POST = get, post
        opened = []

        def open_page(look):
            opened.append(look)
            run = str(len(opened))
            profile = Path(tempfile.mkdtemp(prefix="browser_"))
            try:
                with tempfile.TemporaryFile() as log:
                    process = sheet.launch(browser, profile, f"http://127.0.0.1:{port}/?look={look}&run={run}",
                                           "--enable-logging=stderr", "--v=0", log=log)
                    try:
                        with came:
                            came.wait_for(lambda: run in drawn, timeout=120)
                    finally:
                        sheet.close(process)
                    log.seek(0)
                    said = log.read().decode("utf-8", errors="replace")
            finally:
                # every process of the browser is gone (sheet.close): what still holds the profile after a few
                # seconds is a browser process close() didn't reach, and is said, not left running
                for wait in (0.2, 0.5, 1, 2, 4, 0):
                    try:
                        shutil.rmtree(profile)
                        break
                    except FileNotFoundError:
                        break
                    except OSError:
                        if not wait:
                            raise
                        time.sleep(wait)
            self.assertIn(run, drawn, f"the {look} page never said it was drawn ({browser}):\n{said[-3000:]}")
            return SimpleNamespace(stdout=drawn[run], stderr=said)
        # the app has the one look, the arcade one (the plain look it had before could be chosen on Setup
        # until 2026-10-12): as it opens, and for an address or a stored choice that still asks for the other
        pages = {}
        try:
            for look in ("arcade", "classic"):
                pages[look] = open_page(look)
            # ... and one without a Blender opens on Setup, with the line and its button drawn
            missing = {"name": "Blender", "state": "bad", "text": "Blender 4.2 or later wasn't found", "fix": "", "action": "get_blender"}
            engine.setup = lambda given=None: {**ready(given), "ready": False,
                                               "checks": [missing, *(c for c in ready(given)["checks"] if c["name"] != "Blender")]}
            first = open_page("classic")
            # ... and one with everything, started for the first time, opens on Setup once and says so
            before = len(welcomes)
            engine.setup = lambda given=None: {**ready(given), "welcomed": False}
            new = open_page("classic")
            greeted = welcomes[before:]
        finally:
            served.shutdown()
            served.server_close()
            engine.setup, engine.welcome = real_setup, real_welcome
            server.Handler.do_GET, server.Handler.do_POST = real_get, real_post
        self.assertEqual(welcomes[:before], [], "a first start said by a page that wasn't one")
        self.assertEqual(greeted, [True])
        # (the page on show; "fresh" for its first second, while the look slides it in)
        on_setup = r'class="page on(?: fresh)?" id="setup"'
        self.assertRegex(new.stdout, on_setup)
        self.assertNotIn('id="start-go" hidden', new.stdout)
        self.assertEqual([line for line in first.stderr.splitlines() if "CONSOLE" in line and "Uncaught" in line], [])
        self.assertRegex(first.stdout, on_setup)
        for mark in ("Blender 4.2 or later wasn't found", "Get Blender", "App version"):
            self.assertTrue(mark in first.stdout, f"first start: {mark}")
        self.assertNotRegex(first.stdout, r'class="page on[^"]*" id="home"')
        self.assertNotRegex(new.stdout, r'class="page on[^"]*" id="home"')
        for look, done in pages.items():
            page = done.stdout
            errors_said = [line for line in done.stderr.splitlines() if "CONSOLE" in line and ("Uncaught" in line or "SyntaxError" in line)]
            self.assertEqual(errors_said, [], look)
            # each is drawn by its module's last lines, or by main.js after every import ran
            # (quick.js and guide.js too: the automatic mode's switch, and what to check on each step's page)
            for mark in ('data-tool="brush"', "New character", 'class="page on" id="home"', "Open or create a character first",
                         'id="a-mode"', "Mode: automatic", "What to check here"):
                self.assertTrue(mark in page, f"not drawn in the {look} look: {mark}")
            # the arcade look's stage, roster and menus are there, whichever was asked for
            for mark in ('<html lang="en" class="arcade', 'id="a-bg"', 'class="slot new', 'id="menu"', 'id="a-crumbs"'):
                self.assertTrue(mark in page, f"asked for the {look} look: {mark}")
            # ... and Setup offers no other: its switches are of motion, sound and music
            for mark in ('id="look"', "Classic"):
                self.assertFalse(mark in page, f"asked for the {look} look: {mark}")
            for mark in ('id="look-still"', 'id="look-sound"', 'id="look-music"'):
                self.assertTrue(mark in page, f"asked for the {look} look: {mark}")

    def test_what_is_chosen_by_a_click_can_be_chosen_from_the_keyboard(self):
        # rows of steps, pieces, joints and tested pieces, and the headings the pieces are sorted by, were a
        # mouse's alone: the tab key passed over them (at every width; seen in the pages 2026-10-12)
        import re
        script = lambda name: (server.GUI / "js" / name).read_text(encoding="utf-8")  # noqa: E731
        shared = script("ui.js")
        self.assertIn("export function keyable(el) { el.tabIndex = 0; }", shared)
        self.assertIn('el.matches?.("tr[tabindex], th[tabindex]")', shared)
        for name, rows in (("build.js", 1), ("parts.js", 2), ("joints.js", 1), ("movement.js", 1)):
            page = script(name)
            # ... every row a page gives a click to, it gives the keyboard too (and the import)
            self.assertEqual(len(re.findall(r"\bkeyable\b", page)), rows + 1, name)
            self.assertEqual(len(re.findall(r"\btr\.onclick = ", page)), 1, name)
        # the buttons of a row wrap where a list is narrow: held on one line, the last was cut off at 1280 px
        self.assertNotIn("nowrap", script("build.js"))
        self.assertIn("#piece-rows td:last-child .act { white-space: normal; }", (server.GUI / "app.css").read_text(encoding="utf-8"))

    def test_a_build_says_how_long_its_steps_took_the_last_time(self):
        # "a few minutes", whatever the character: its last build's own seconds are there to be added up
        script = lambda name: (server.GUI / "js" / name).read_text(encoding="utf-8")  # noqa: E731
        self.assertIn("export function tookLast(status, from)", script("core.js"))
        # said in the band over the steps, and of the steps to build again in "What's left"
        self.assertIn("tookLast(st)", script("look.js"))
        self.assertIn("tookLast(S.status, next.from)", script("build.js"))
        # the sum itself, as the page works it out (Node runs the module's one function, where there is a Node)
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("no Node on this computer to run the page's own function")
        source = script("core.js").replace("\r\n", "\n")
        body = source[source.index("export function tookLast"):]
        body = body[:body.index("\n}\n") + 3].replace("export function", "function")
        steps = lambda *rows: json.dumps({"steps": [{"step": name, "runs": runs, "seconds": seconds} for name, runs, seconds in rows]})  # noqa: E731
        asked = [(steps(("survey", True, 4), ("build", True, 94), ("hair", True, 6), ("chest", False, None)), "null"),   # 104 s
                 (steps(("survey", True, 4), ("build", True, 94), ("hair", True, 6)), '"hair"'),                           # 6 s: never under 10
                 (steps(("survey", True, 4), ("build", True, None)), "null"),                                              # one never run
                 (steps(("survey", True, 61), ("build", True, 59)), "null"),                                               # 120 s
                 (steps(("survey", True, 4)), '"export"')]                                                                  # no such step
        program = body + "\nconsole.log(JSON.stringify([" + ", ".join(f"tookLast({status}, {start})" for status, start in asked) + "]));"
        said = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(said.stdout or "null"), ["1 min 40 s", "10 s", None, "2 min", None], said.stderr[-500:])


class Detector(unittest.TestCase):
    """The face detector's own Python, made by the Setup page's button (face_guess.install)."""

    def setUp(self):
        from app import face_guess
        self.guess = face_guess
        self.folder = tempfile.TemporaryDirectory()
        self.home = Path(self.folder.name) / "tools" / "face_detector"
        self.real = face_guess.base_python, face_guess._has_detector
        face_guess.forget()

    def tearDown(self):
        self.guess.base_python, self.guess._has_detector = self.real
        self.guess.forget()
        self.folder.cleanup()

    def runner(self, codes, seen):
        class Process:
            def __init__(self, command, **kwargs):
                seen.append(command)
                self.returncode = codes.pop(0)
                kwargs["stdout"].write("pip said something\n")

            def poll(self):
                return self.returncode

            def kill(self):
                pass

            def wait(self):
                pass
        return Process

    def test_it_makes_its_own_python_and_installs_into_it(self):
        self.guess.base_python = lambda: "py312"
        self.guess._has_detector = lambda python: True
        seen, events = [], []
        made = self.guess.install(on_event=events.append, home=self.home, run=self.runner([0, 0, 0], seen), free=lambda path: 50 << 30)
        own = str(self.guess.own_python(self.home))
        self.assertEqual(made["python"], own)
        self.assertEqual(seen[0], ["py312", "-m", "venv", str(self.home)])
        self.assertEqual(seen[1][:5] + seen[1][-2:], [own, "-m", "pip", "install", "torch", "--index-url", self.guess.TORCH_INDEX])
        self.assertEqual(seen[2], [own, "-m", "pip", "install", self.guess.PACKAGE])
        self.assertEqual([e["status"] for e in events], ["running", "done"] * 3)
        self.assertTrue(Path(events[0]["log"]).read_text().count("pip said something") == 3)

    def test_it_says_what_stops_it(self):
        self.guess.base_python = lambda: None
        with self.assertRaisesRegex(self.guess.NoDetector, "Python 3.12 or newer"):
            self.guess.install(home=self.home)
        self.guess.base_python = lambda: "py312"
        with self.assertRaisesRegex(self.guess.NoDetector, "GB free"):
            self.guess.install(home=self.home, free=lambda path: 1 << 30)
        seen = []
        with self.assertRaisesRegex(self.guess.NoDetector, "downloading PyTorch.*failed"):
            self.guess.install(home=self.home, run=self.runner([0, 1], seen), free=lambda path: 50 << 30)
        self.assertEqual(len(seen), 2)  # the detector isn't tried after PyTorch failed
        with self.assertRaisesRegex(self.guess.NoDetector, "stopped"):
            class Slow(self.runner([None], [])):
                pass
            self.guess.install(home=self.home, run=Slow, cancelled=lambda: True, free=lambda path: 50 << 30)

    def test_an_older_python_is_passed_over_for_one_the_launcher_lists(self):
        import os
        import shutil
        listed = chr(10).join((" -V:3.14 *        C:/new/python.exe", " -V:3.12          C:/mid/python.exe", " -V:3.10          C:/old/python.exe"))
        which, is_file = shutil.which, Path.is_file
        try:
            # the app on Python 3.10, on Windows, with the launcher's three Pythons
            self.guess.sys = SimpleNamespace(version_info=(3, 10, 6), executable="old")
            self.guess.os = SimpleNamespace(name="nt", environ={})
            shutil.which = lambda name: "py" if name == "py" else None
            Path.is_file = lambda path: True
            found = self.guess.base_python(run=lambda *a, **k: SimpleNamespace(stdout=listed))
        finally:
            self.guess.sys, self.guess.os, shutil.which, Path.is_file = sys, os, which, is_file
        self.assertEqual(found, "C:/mid/python.exe")  # the oldest that will do, never the one too old


class Vendor(unittest.TestCase):
    """The app's own copies of what its pages load from the internet, in a folder of the test's."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.real = vendor.FOLDER, vendor.PACKAGES
        vendor.FOLDER = Path(self.folder.name)
        self.files = {"https://cdn.example/npm/lib@1.0.0/a.js": b"export const a = 1;", "https://cdn.example/npm/lib@1.0.0/sub/b.js": b"export const b = 2;",
                      "https://models.example/tracker/1/model.task": b"\0model"}
        import hashlib
        pin = lambda url: hashlib.sha256(self.files[url]).hexdigest()  # noqa: E731
        vendor.PACKAGES = [("views", "https://cdn.example/npm/lib@1.0.0/", {"a.js": pin("https://cdn.example/npm/lib@1.0.0/a.js"),
                                                                            "sub/b.js": pin("https://cdn.example/npm/lib@1.0.0/sub/b.js")}),
                           ("tracker", "https://models.example/tracker/1/", {"model.task": pin("https://models.example/tracker/1/model.task")})]

    def tearDown(self):
        vendor.FOLDER, vendor.PACKAGES = self.real
        self.folder.cleanup()

    def opener(self, served=None):
        import io
        served = self.files if served is None else served
        asked = []

        def open_url(request, timeout=None):
            asked.append(request.full_url)
            return io.BytesIO(served[request.full_url])
        return open_url, asked

    def test_pages_name_the_kept_copy_one_package_at_a_time(self):
        page = '"lib": "https://cdn.example/npm/lib@1.0.0/a.js", MODEL = "https://models.example/tracker/1/model.task"'
        self.assertEqual(vendor.local(page), page)  # nothing kept: the internet's
        open_url, asked = self.opener()
        self.assertEqual(vendor.status()["views"], False)
        done = vendor.fetch(open_url=open_url)
        self.assertEqual((done["views"], done["tracker"], len(done["fetched"]), len(asked)), (True, True, 3, 3))
        self.assertEqual(vendor.local(page), '"lib": "/gui/vendor/npm/lib@1.0.0/a.js", MODEL = "/gui/vendor/tracker/1/model.task"')
        self.assertEqual(vendor.fetch(open_url=open_url)["fetched"], [])  # kept: nothing is asked for again
        (vendor.FOLDER / "tracker" / "1" / "model.task").unlink()
        engine.CANCEL.set()
        try:
            with self.assertRaises(engine.Cancelled):  # Stop reaches the download, and leaves no half file
                vendor.fetch(open_url=open_url)
        finally:
            engine.CANCEL.clear()
        self.assertEqual([p.name for p in (vendor.FOLDER / "tracker" / "1").iterdir()], [])
        vendor.fetch(open_url=open_url)
        (vendor.FOLDER / "tracker" / "1" / "model.task").unlink()
        # the views alone: the tracker's address must stay the internet's, or the webcam test would find nothing
        self.assertEqual(vendor.local(page), '"lib": "/gui/vendor/npm/lib@1.0.0/a.js", MODEL = "https://models.example/tracker/1/model.task"')

    def test_a_download_that_isnt_the_pinned_file_is_refused_and_nothing_kept(self):
        changed = dict(self.files, **{"https://cdn.example/npm/lib@1.0.0/sub/b.js": b"export const b = 'something else';"})
        open_url, _ = self.opener(changed)
        with self.assertRaisesRegex(ValueError, "isn't the file this app was tested with"):
            vendor.fetch(open_url=open_url)
        kept = sorted(p.name for p in vendor.FOLDER.rglob("*") if p.is_file())
        self.assertEqual(kept, ["a.js"])  # the good one stays, the other left neither itself nor a .part
        self.assertEqual(vendor.local('"https://cdn.example/npm/lib@1.0.0/a.js"'), '"https://cdn.example/npm/lib@1.0.0/a.js"')

    def test_the_copies_in_the_repository_are_the_pinned_files(self):
        # byte for byte: a line-ending conversion on checkout (.gitattributes keeps them out) or an
        # edited copy would otherwise go unnoticed, since the pages only ask whether a file is there
        import hashlib
        vendor.FOLDER, vendor.PACKAGES = self.real
        for _, address, files in vendor.PACKAGES:
            for name, pinned in files.items():
                path = vendor.FOLDER / vendor._under(address) / name
                with self.subTest(file=name):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), pinned)
        self.assertEqual((vendor.status()["views"], vendor.status()["tracker"]), (True, True))

    def test_every_three_file_the_pages_import_is_pinned(self):
        # with a copy kept, three/addons/ resolves here: an addon a page imports and this list lacks would be a 404
        import re
        vendor.FOLDER, vendor.PACKAGES = self.real
        pinned = next(files for _, address, files in vendor.PACKAGES if address.endswith("/three@0.180.0/"))
        pages = [*(REPO / "app" / "gui" / "js").glob("*.js"), *(REPO / "viewer").glob("*.html"), REPO / "app" / "gui" / "index.html"]
        imported = {match for page in pages for match in re.findall(r"three/addons/([\w/]+\.js)", page.read_text(encoding="utf-8"))}
        self.assertTrue(imported)
        self.assertEqual([name for name in sorted(imported) if f"examples/jsm/{name}" not in pinned], [])
        # and the addresses the pages use are the pinned ones, version for version
        text = (REPO / "viewer" / "index.html").read_text(encoding="utf-8") + (REPO / "app" / "gui" / "index.html").read_text(encoding="utf-8")
        for _, address, _ in vendor.PACKAGES:
            self.assertIn(address, text)


class Inside(unittest.TestCase):
    """What shows through a hole cut in the exported file's surface, as apps draw it (app/inside.py)."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def page_said(self, deep, hair=0.0, by="EyeLids"):
        """A report as viewer/inside.html sends it: three cameras, the middle one seeing `deep` mm2 over the limit."""
        shot = lambda label, mm2: {"label": label, "from": [0, 0], "pose": "rest", "model": 90000, "window_mm2": 1600.0, "deep_mm2": mm2,  # noqa: E731
                                   "through_mm2": 0.0, "behind90_mm": 6.0, "by": {by: 10} if mm2 else {}, "hair_inside_mm2": 0.0, "head_inside_mm2": 0.0}
        return {"cut_out": ["EyeIris", "EyeLids"], "deep_mm": 12, "face": [shot("face -20,0", deep / 3), shot("face 0,0", deep), shot("face 20,0", 0.0)],
                "hair": [{**shot("rest 80,-10", 0.0), "hair_inside_mm2": hair}]}

    def test_a_deep_white_fails_and_a_shallow_one_does_not(self):
        from app import inside
        # the worst camera's counts of the fox girl's deep white and of one eye of it, the catgirl's, and the
        # most any good build read (the plain boy's white 6 mm down)
        for deep, fails in ((543.8, True), (277.7, True), (408.3, True), (11.6, False), (0.0, False), (inside.LIMIT_MM2, False)):
            found = inside.judge(self.page_said(deep))
            self.assertEqual(bool(found["failed"]), fails, deep)
            self.assertEqual((found["deep_mm2"], found["shot"]), (deep, "face 0,0") if deep else (0.0, "face -20,0"))
        line = inside.judge(self.page_said(408.3))["failed"][0]
        self.assertIn("408 mm2", line)
        self.assertIn("EyeLids", line)
        # the hair's inside beside the neck is measured and never judged: every long-haired build reads thousands
        found = inside.judge(self.page_said(0.0, hair=4911.0))
        self.assertEqual((found["hair_inside_mm2"], found["hair_shot"], found["failed"]), (4911.0, "rest 80,-10", []))

    def test_no_browser_is_started_for_what_is_no_vrm_and_a_build_is_never_lost_to_it(self):
        from unittest import mock
        from app import inside, sheet
        fake = self.root / "a.vrm"
        fake.write_bytes(b"glTF-model")
        with mock.patch.object(sheet, "launch", side_effect=AssertionError("a browser was started")):
            self.assertEqual(inside.check(fake, self.root / "seen"), {"left": "not a VRM file"})
            self.assertEqual(inside.check(self.root / "missing.vrm", self.root / "seen"), {"left": "not a VRM file"})
        real = Engine.write_glb(self, "real.vrm", {"asset": {"version": "2.0"}, "extensions": {"VRMC_vrm": {}}})
        for error, said, code in ((sheet.NoBrowser("no Edge or Chrome found"), "no Edge or Chrome found", "VM-NO-BROWSER"),
                                  (TimeoutError("inside.html wasn't drawn in 300 s"), "couldn't be drawn", "VM-NOT-DRAWN"),
                                  (RuntimeError("this browser's graphics can't be read back as numbers"), "can't be read back", "VM-NOT-DRAWN"),
                                  (RuntimeError("a texture didn't load (the browser's profile drive may be full)"), "a texture didn't load", "VM-NOT-DRAWN")):
            with mock.patch.object(sheet, "browse", side_effect=error):
                left = inside.check(real, self.root / "seen")
                self.assertIn(said, left["left"])
                self.assertEqual(left["code"], code)
                # the room the browser had is said beside a file it drew nothing of
                self.assertEqual("GB free where the browser kept its profile" in left["left"], code == "VM-NOT-DRAWN")

    def test_the_exported_files_check_ends_with_it_and_the_verdict_takes_it_in(self):
        from unittest import mock
        from app import inside
        data = manifest.new("model.glb", "Test")
        data["out_dir"] = "out"
        manifest.save(data, self.root / "character.json")
        real = engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS

        def blender(blender, blend_in, args, log, extra):
            Path(log).write_text("checked\n")
            Path(args[args.index("--json") + 1]).write_text(json.dumps({"failed": [], "poses": {}, "unweighted": 0}))
            return 0
        engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS = blender, lambda path: 100 << 30, lambda: 16 << 30, self.root / "settings.json"
        try:
            project = Project(self.root / "character.json", blender="blender")
            vrm = Engine.write_glb(self, "14_export.vrm", Engine.avatar(self))
            project._record("export", {"status": "done", "vrm": str(vrm)})
            seen = inside.judge(self.page_said(408.3))
            with mock.patch.object(inside, "check", return_value=seen) as checked:
                entry = project._run_step("vrm_check", engine.NAMES.index("vrm_check"), self.root)
            self.assertEqual(Path(checked.call_args.args[0]), vrm)
            self.assertEqual(entry["report"]["seen"], seen)
            # in the check's own report, where `compare` and `regress` read it, and in the step's log
            self.assertEqual(json.loads(Path(entry["json"]).read_text())["seen"]["deep_mm2"], 408.3)
            self.assertIn("seen: ", Path(entry["log"]).read_text())
            project._record("vrm_check", entry)
            verdict = project.verdict([])
            self.assertFalse(verdict["ready"])
            self.assertTrue(any(page == "review" and line.startswith("as apps draw it: 408 mm2") for page, line in verdict["issues"]), verdict)
            # with no browser, a file that couldn't be drawn is a look, never an issue
            project.state["steps"]["vrm_check"]["report"]["seen"] = {"left": "no Edge or Chrome found", "code": "VM-NO-BROWSER"}
            verdict = project.verdict([])
            self.assertFalse(any("as apps draw it" in line or "wasn't drawn" in line for _, line in verdict["issues"]))
            self.assertTrue(any("wasn't drawn" in line and "no Edge or Chrome" in line for _, line in verdict["looks"]), verdict)
            # ... but a browser that drew nothing left the eyes unlooked at: the fox girl's moving iris over a hole
            # passed so on an E: with 4 GB free. Not ready, and the way on said
            undrawn = {"left": "the file couldn't be drawn: a texture didn't load (4.0 GB free where the browser kept its profile)",
                       "code": "VM-NOT-DRAWN"}
            project.state["steps"]["vrm_check"]["report"]["seen"] = undrawn
            verdict = project.verdict([])
            line = next(line for page, line in verdict["issues"] if page == "review" and "wasn't drawn" in line)
            self.assertIn("a texture didn't load", line)
            self.assertIn("free space on the drive", line)
            self.assertFalse(verdict["ready"])
            self.assertFalse(any("wasn't drawn" in line for _, line in verdict["looks"]))
            # said as the step ends too
            with mock.patch.object(inside, "check", return_value=undrawn):
                entry = project._run_step("vrm_check", engine.NAMES.index("vrm_check"), self.root)
            self.assertTrue(any("a texture didn't load" in warning and "weren't looked at" in warning for warning in entry["warnings"]), entry)
        finally:
            engine._blender, engine.free_disk, engine.free_memory, engine.SETTINGS = real

    def test_a_vrm_taken_in_keeps_what_shows_through_it_as_a_look(self):
        from app import inside
        Engine.write_glb(self, "avatar.vrm", Engine.avatar(self))
        data = manifest.new("avatar.vrm", "Test")
        data["out_dir"] = "out"
        manifest.save(data, self.root / "character.json")
        real, engine.SETTINGS = engine.SETTINGS, self.root / "settings.json"
        try:
            project = Project(self.root / "character.json", blender="blender")
            vrm = Engine.write_glb(self, "14_export.vrm", Engine.avatar(self))
            project.state["steps"]["export"] = {"status": "done", "vrm": str(vrm)}
            project.state["steps"]["vrm_check"] = {"status": "done", "report": {"ships": engine.shipped(vrm), "seen": inside.judge(self.page_said(408.3))}}
            verdict = project.verdict([])
            self.assertFalse(any("as apps draw it" in line for _, line in verdict["issues"]))
            self.assertTrue(any("as apps draw it: 408 mm2" in line for _, line in verdict["looks"]), verdict)
        finally:
            engine.SETTINGS = real

    def test_two_builds_are_compared_by_it(self):
        check = lambda deep: {"poses": {}, "unweighted": 0, **({"seen": {"deep_mm2": deep}} if deep is not None else {})}  # noqa: E731
        worse, better = engine.compare_checks.changes(check(0.6), check(408.3))
        self.assertEqual((worse, better), (["mm2 seen deep through a hole in the surface 0.6 -> 408.3"], []))
        worse, better = engine.compare_checks.changes(check(543.8), check(0.0))
        self.assertEqual((worse, better), ([], ["mm2 seen deep through a hole in the surface 543.8 -> 0"]))
        # the same file drawn again, and a report from before the count was kept
        self.assertEqual(engine.compare_checks.changes(check(0.6), check(1.8)), ([], []))
        self.assertEqual(engine.compare_checks.changes(check(None), check(408.3)), ([], []))
        # a build not drawn after one that was says nothing of its holes: worse, not the same (the fox girl's)
        undrawn = {"poses": {}, "unweighted": 0, "seen": {"left": "the file couldn't be drawn: a texture didn't load", "code": "VM-NOT-DRAWN"}}
        worse, better = engine.compare_checks.changes(check(0.6), undrawn)
        self.assertEqual((len(worse), better), (1, []))
        self.assertIn("not drawn as apps draw it", worse[0])
        self.assertIn("a texture didn't load", worse[0])
        self.assertEqual(engine.compare_checks.changes(undrawn, check(0.6)), ([], ["drawn as apps draw it, the build before wasn't"]))
        self.assertEqual(engine.compare_checks.changes(check(None), undrawn), ([], []))

    def tiny_vrm(self, name, behind, patch=False):
        """A VRM 1.0 file of three sheets at head height, facing forward: an eye's frame (material EyeLids, cut
        out: opaque with a 4 x 3 cm window in its picture), a surface `behind` metres behind the frame, and
        with `patch` a blended sheet with the same hole 3 cm in front of both."""
        import zlib
        chunk = lambda kind, body: struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))  # noqa: E731
        rows = b"".join(b"\0" + b"".join(b"\xe0\xb0\xa0" + (b"\0" if 4 <= x < 12 and 4 <= y < 12 else b"\xff") for x in range(16)) for y in range(16))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
        bones = ["hips", "spine", "neck", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
                 "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]
        heights = {"hips": 0.9, "spine": 1.0, "neck": 1.35, "head": 1.43}
        # (half width, half height, depth, material) of each sheet, centred where the page aims: 7 cm over the head joint
        sheets = [(0.04, 0.03, 0.002, 1), (0.15, 0.15, -behind, 0)] + ([(0.04, 0.03, 0.03, 2)] if patch else [])
        binary, views, accessors, primitives = b"", [], [], []

        def add(body, **accessor):
            nonlocal binary
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(body)})
            accessors.append({"bufferView": len(views) - 1, "componentType": 5126, **accessor})
            binary += body + b"\0" * (-len(body) % 4)
            return len(accessors) - 1
        for w, h, z, material in sheets:
            corners = [(-w, 1.5 - h, z), (w, 1.5 - h, z), (w, 1.5 + h, z), (-w, 1.5 + h, z)]
            position = add(struct.pack("<12f", *(c for corner in corners for c in corner)), count=4, type="VEC3",
                           min=[-w, 1.5 - h, z], max=[w, 1.5 + h, z])
            uv = add(struct.pack("<8f", 0, 1, 1, 1, 1, 0, 0, 0), count=4, type="VEC2")
            index = add(struct.pack("<6H", 0, 1, 2, 0, 2, 3), count=6, type="SCALAR")
            accessors[index]["componentType"] = 5123
            primitives.append({"attributes": {"POSITION": position, "TEXCOORD_0": uv}, "indices": index, "material": material})
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(png)})
        binary += png
        picture = {"baseColorTexture": {"index": 0}}
        gltf = {"asset": {"version": "2.0"}, "extensionsUsed": ["VRMC_vrm"], "scene": 0, "scenes": [{"nodes": [0, len(bones)]}],
                "nodes": [{"name": bones[0], "translation": [0, heights["hips"], 0], "children": list(range(1, len(bones)))}]
                + [{"name": bone, "translation": [0, heights.get(bone, 1.0) - heights["hips"], 0]} for bone in bones[1:]] + [{"name": "Body", "mesh": 0}],
                "meshes": [{"primitives": primitives}],
                "materials": [{"name": "Skin", "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.9, 1]}},
                              {"name": "EyeLids", "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": 0.5, "pbrMetallicRoughness": picture},
                              {"name": "EyePaint", "doubleSided": True, "alphaMode": "BLEND", "pbrMetallicRoughness": picture}],
                "textures": [{"source": 0, "sampler": 0}], "samplers": [{"magFilter": 9728, "minFilter": 9728}],
                "images": [{"bufferView": len(views) - 1, "mimeType": "image/png"}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
                "extensions": {"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Test", "authors": ["test"], "licenseUrl": "https://vrm.dev/licenses/1.0/"},
                                            "humanoid": {"humanBones": {bone: {"node": i} for i, bone in enumerate(bones)}}}}}
        return Engine.write_glb(self, name, gltf, binary)

    def test_a_browser_counts_what_shows_through_a_cut_window_and_not_through_a_blended_patch(self):
        from app import inside, sheet
        try:
            sheet.find_browser()
        except sheet.NoBrowser as error:
            self.skipTest(str(error))
        # the window's 1,200 mm2 with a surface 3 cm behind it, as the fox girl's socket lay
        deep = inside.check(self.tiny_vrm("deep.vrm", 0.03), self.root / "deep")
        self.assertNotIn("left", deep, deep)
        self.assertEqual(deep["cut_out"], ["EyeLids"])
        self.assertTrue(900 < deep["deep_mm2"] < 1500 and deep["shot"] == "face 0,0", deep)
        self.assertEqual(len(deep["failed"]), 1)
        self.assertTrue((self.root / "deep" / "inside_worst.png").stat().st_size > 1000)
        # the same window with the surface 3 mm behind, under a blended patch standing 3 cm off it (model B's
        # eye patches stand 13 mm off her face): holes in a blended sheet are no window
        shallow = inside.check(self.tiny_vrm("shallow.vrm", 0.003, patch=True), self.root / "shallow")
        self.assertEqual((shallow.get("deep_mm2"), shallow.get("failed")), (0.0, []), shallow)
        self.assertGreater(shallow["window_mm2"], 300)  # the frame's own window is still read where the patch leaves it clear

    def test_an_eye_layer_past_the_face_fails_and_one_tucked_behind_it_does_not(self):
        from app import inside
        # the worst camera of the miko's far-eye panel before 326d777, and the most any good build read
        for past, fails in ((74.5, True), (8.5, False), (inside.PAST_MM2, False)):
            said = self.page_said(0.0)
            said["face"][1]["past_mm2"] = past
            found = inside.judge(said)
            self.assertEqual(bool(found["failed"]), fails, past)
            self.assertEqual((found["past_mm2"], found["past_shot"]), (past, "face 0,0"))
        self.assertIn("74 mm2 of an eye layer", inside.judge({**said, "face": [{**said["face"][1], "past_mm2": 74.5}]})["failed"][0])
        # a report from before the eye layers were counted
        self.assertEqual((inside.judge(self.page_said(0.0))["past_mm2"], inside.judge(self.page_said(0.0))["failed"]), (0.0, []))

    def layered_vrm(self, name, grid_at):
        """A VRM 1.0 file of a face (material face_patch, 12 x 12 cm), a white sheet as eyes.py makes it (a
        FRAME_GRID of quads in the face's picture, 3 x 6 cm, its lower left corner at `grid_at` (x, z)) and a
        sheet of hair 5 cm behind the face on a spring: skinned, the face and sheet to the head."""
        cols, rows = 24, 16
        bones = ["hips", "spine", "neck", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
                 "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]
        heights = {"hips": 0.9, "spine": 1.0, "neck": 1.35, "head": 1.43}
        world = [heights.get(bone, 1.0) for bone in bones] + [1.6, 1.4]  # and the hair's two joints
        hair, tip = len(bones), len(bones) + 1
        binary, views, accessors, primitives = b"", [], [], []

        def add(body, component=5126, **accessor):
            nonlocal binary
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(body)})
            accessors.append({"bufferView": len(views) - 1, "componentType": component, **accessor})
            binary += body + b"\0" * (-len(body) % 4)
            return len(accessors) - 1

        def sheet(x0, y0, x1, y1, z, columns, rows_, material, joint):
            points = [(x0 + (x1 - x0) * i / columns, y0 + (y1 - y0) * j / rows_, z) for j in range(rows_ + 1) for i in range(columns + 1)]
            quads = [(j * (columns + 1) + i, j * (columns + 1) + i + 1, (j + 1) * (columns + 1) + i + 1, (j + 1) * (columns + 1) + i)
                     for j in range(rows_) for i in range(columns)]
            triangles = [k for a, b, c, d in quads for k in (a, b, c, a, c, d)]
            position = add(struct.pack(f"<{3 * len(points)}f", *(c for p in points for c in p)), count=len(points), type="VEC3",
                           min=[min(p[k] for p in points) for k in range(3)], max=[max(p[k] for p in points) for k in range(3)])
            joints = add(struct.pack(f"<{4 * len(points)}B", *([joint, 0, 0, 0] * len(points))), 5121, count=len(points), type="VEC4")
            weights = add(struct.pack(f"<{4 * len(points)}f", *([1.0, 0, 0, 0] * len(points))), count=len(points), type="VEC4")
            index = add(struct.pack(f"<{len(triangles)}H", *triangles), 5123, count=len(triangles), type="SCALAR")
            primitives.append({"attributes": {"POSITION": position, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": index, "material": material})
        head = bones.index("head")
        sheet(-0.06, 1.44, 0.06, 1.56, 0.0, 6, 6, 0, head)
        sheet(grid_at[0], 1.47, grid_at[0] + 0.03, 1.53, grid_at[1], cols, rows, 0, head)
        sheet(-0.15, 1.35, 0.15, 1.65, -0.05, 6, 6, 1, hair)
        binds = add(struct.pack(f"<{16 * len(world)}f", *(v for h in world for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -h, 0, 1))), count=len(world), type="MAT4")
        nodes = ([{"name": bones[0], "translation": [0, heights["hips"], 0], "children": list(range(1, len(bones)))}]
                 + [{"name": bone, "translation": [0, heights.get(bone, 1.0) - heights["hips"], 0]} for bone in bones[1:]])
        nodes[head]["children"] = [hair]
        nodes += [{"name": "hair", "translation": [0, 1.6 - heights["head"], 0], "children": [tip]}, {"name": "hairTip", "translation": [0, -0.2, 0]},
                  {"name": "Body", "mesh": 0, "skin": 0}]
        gltf = {"asset": {"version": "2.0"}, "extensionsUsed": ["VRMC_vrm", "VRMC_springBone"], "scene": 0, "scenes": [{"nodes": [0, len(nodes) - 1]}],
                "nodes": nodes, "meshes": [{"primitives": primitives}], "skins": [{"joints": list(range(len(world))), "inverseBindMatrices": binds}],
                "materials": [{"name": "face_patch", "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.95, 0.8, 0.7, 1]}},
                              {"name": "Hair", "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.1, 0.15, 1]}}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
                "extensions": {"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Test", "authors": ["test"], "licenseUrl": "https://vrm.dev/licenses/1.0/"},
                                            "humanoid": {"humanBones": {bone: {"node": i} for i, bone in enumerate(bones)}}},
                               "VRMC_springBone": {"specVersion": "1.0", "springs": [{"joints": [{"node": hair, "stiffness": 1.0, "gravityPower": 0, "dragForce": 0.5},
                                                                                                 {"node": tip}]}]}}}
        return Engine.write_glb(self, name, gltf, binary)

    def test_a_browser_finds_an_eye_layer_standing_past_the_face(self):
        from app import inside, sheet
        try:
            sheet.find_browser()
        except sheet.NoBrowser as error:
            self.skipTest(str(error))
        # the sheet beside the face's edge, before the hair: the miko's far-eye panel, a strip of skin from the side
        past = inside.check(self.layered_vrm("past.vrm", (0.065, 0.0)), self.root / "past")
        self.assertNotIn("left", past, past)
        self.assertGreater(past["past_mm2"], 10 * inside.PAST_MM2, past)
        self.assertTrue(any("eye layer" in line for line in past["failed"]), past)
        # tucked 1 cm behind the middle of the face: never seen first
        tucked = inside.check(self.layered_vrm("tucked.vrm", (-0.015, -0.01)), self.root / "tucked")
        self.assertEqual((tucked.get("past_mm2"), tucked.get("failed")), (0.0, []), tucked)
        report = json.loads((self.root / "tucked" / "inside.json").read_text())
        self.assertEqual(report["eye_layer_vertices"], 25 * 17)

    def test_the_page_knows_the_white_sheet_by_the_grid_eyes_py_makes(self):
        import re
        from app import inside
        made = re.search(r"^FRAME_GRID = \((\d+), (\d+)\)", (REPO / "addon" / "vtuber_maker" / "pipeline" / "eyes.py").read_text(encoding="utf-8"), re.M)
        known = re.search(r"^const FRAME_GRID = \[(\d+), (\d+)\];", inside.PAGE.read_text(encoding="utf-8"), re.M)
        self.assertEqual(made.groups(), known.groups())


class Blenders(unittest.TestCase):
    """One Blender kept through a run for its steps (engine._Worker), and the build's last steps side by side
    when the computer has room (Project._side_by_side): what the engine does with both. That the files they
    leave are the ones a Blender a step leaves was held byte for byte on real builds (SETUP.md "One Blender
    for a build's steps; the last steps side by side"); scripts/_worker.py itself is tested in Blender
    (tests/test_worker.py)."""

    def setUp(self):
        self.made = Engine("test_survey_decides_what_was_left_to_it")
        self.made.setUp()
        self.root = self.made.root
        self.environment = dict(os.environ)
        self.real = engine.subprocess.Popen, engine.time.sleep, engine.LOAD.idle_cores
        self.started, self.told = [], []

    def tearDown(self):
        engine.subprocess.Popen, engine.time.sleep, engine.LOAD.idle_cores = self.real
        engine.CANCEL.clear()
        os.environ.clear()
        os.environ.update(self.environment)
        self.made.tearDown()

    def kept(self, answer=lambda order: 0):
        """A stand-in for the kept Blender: takes each step told on its input and answers it with
        `answer(order)`: the step's exit code, None to die under it, or ... to say nothing at all."""
        test = self

        class Kept:
            returncode = None

            def __init__(self, command, **kwargs):
                test.started.append([str(c) for c in command])
                self.stdin = self

            def write(self, raw):
                order = json.loads(raw)
                test.told.append(order)
                code = answer(order)
                if code is None:
                    self.returncode = 3
                elif code is not ...:
                    with open(order["log"], "a") as log:
                        log.write("did it\n")
                    Path(order["done"]).write_text(json.dumps({"code": code}))

            def flush(self):
                pass

            def close(self):
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9
        return Kept

    def test_a_steps_blender_arguments_become_what_the_kept_blender_runs(self):
        extra = [*engine.NO_EXPORT, "--python-expr", "x = 1", "-P", str(engine.FINGERPRINT), "--python-exit-code", "1"]
        self.assertEqual(engine._orders(extra, "fix_mouth.py"),
                         [["expr", engine.NO_EXPORT[1]], ["expr", "x = 1"], ["file", str(engine.FINGERPRINT)],
                          ["file", str(engine.SCRIPTS / "fix_mouth.py")]])
        # an argument that is neither: that step keeps a Blender of its own
        self.assertIsNone(engine._orders(["--debug-all"], "fix_mouth.py"))
        self.assertTrue(engine.WORKER.exists())

    def test_the_user_can_have_a_blender_a_step_and_one_at_a_time(self):
        for name in ("OC2VRM_ONE_BLENDER", "OC2VRM_BLENDERS"):
            os.environ.pop(name, None)
        self.assertTrue(engine.one_blender())
        self.assertIsNone(engine.blenders_at_once())  # as many as there is room for
        engine._write_json(engine.SETTINGS, {"one_blender": False, "blenders": 2})
        self.assertFalse(engine.one_blender())
        self.assertEqual(engine.blenders_at_once(), 2)
        os.environ["OC2VRM_ONE_BLENDER"], os.environ["OC2VRM_BLENDERS"] = "1", "0"
        self.assertTrue(engine.one_blender())
        self.assertEqual(engine.blenders_at_once(), 1)  # never under one
        # a copy of the app that lacks the kept Blender's script keeps none, whatever is said
        real, engine.WORKER = engine.WORKER, self.root / "no" / "_worker.py"
        try:
            self.assertFalse(engine.one_blender())
        finally:
            engine.WORKER = real

    def test_steps_that_start_from_a_blend_are_told_to_the_one_blender(self):
        engine.subprocess.Popen = self.kept(lambda order: 0 if "good.blend" in order["argv"] else 4)
        worker = engine._Worker("blender", self.root / "run")
        blend, log = self.root / "in.blend", self.root / "run" / "02_hair.log"
        self.assertEqual(engine._blender("blender", blend, ["fix_hair_weights.py", "good.blend"], log, engine.NO_EXPORT, worker=worker), 0)
        self.assertEqual(self.started, [["blender", "-b", "--factory-startup", "-P", str(engine.STARTUP), "-P", str(engine.WORKER)]])
        order = self.told[0]
        self.assertEqual((order["blend"], order["log"]), (str(blend), str(log)))
        self.assertEqual(order["run"], [["expr", engine.NO_EXPORT[1]], ["file", str(engine.SCRIPTS / "fix_hair_weights.py")]])
        # the command line a Blender of the step's own would have had: the scripts read their arguments off it
        self.assertEqual(order["argv"], [str(c) for c in engine._command("blender", blend, ["fix_hair_weights.py", "good.blend"], engine.NO_EXPORT)])
        said = log.read_text()
        self.assertTrue(said.startswith("blender -b --factory-startup") and said.endswith("did it\n"), said)
        self.assertFalse(Path(order["done"]).exists())
        # the same Blender for the next step, whose own exit code comes back
        self.assertEqual(engine._blender("blender", blend, ["add_ahoge_spring.py", "none.blend"], self.root / "run" / "03_ahoge.log", [], worker=worker), 4)
        self.assertEqual(len(self.started), 1)
        # what it can't be told is not started at all: the caller gives that step a Blender of its own
        self.assertIsNone(worker.run(blend, ["fix_mouth.py"], self.root / "run" / "12_mouth.log", ["--debug-all"]))
        self.assertFalse((self.root / "run" / "12_mouth.log").exists())
        worker.close()
        self.assertIsNone(worker.process)

    def test_a_kept_blender_that_dies_fails_its_step_and_the_next_step_starts_another(self):
        answers = iter([None, 0])
        engine.subprocess.Popen = self.kept(lambda order: next(answers))
        worker = engine._Worker("blender", self.root / "run")
        blend, log = self.root / "in.blend", self.root / "run" / "05_loose.log"
        self.assertEqual(worker.run(blend, ["fix_loose_pieces.py"], log, []), 3)
        self.assertIn("Blender ended under the step with exit code 3", log.read_text())
        self.assertEqual(worker.run(blend, ["fix_loose_pieces.py"], log, []), 0)
        self.assertEqual(len(self.started), 2)

    def test_the_kept_blender_is_stopped_as_any_blender_is(self):
        engine.subprocess.Popen = self.kept(lambda order: ...)  # it never answers
        blend = self.root / "in.blend"
        worker = engine._Worker("blender", self.root / "run")
        engine.time.sleep = lambda seconds: engine.CANCEL.set()
        log = self.root / "run" / "09_tpose.log"
        self.assertEqual(worker.run(blend, ["to_tpose.py"], log, []), engine.CANCELLED)
        self.assertIn("CANCELLED by the user", log.read_text())
        self.assertIsNone(worker.process)  # ended: nothing of the run outlives it
        engine.CANCEL.clear()
        # ... and when the memory runs out under it, with the words the page knows
        engine.time.sleep = lambda seconds: None
        engine.free_memory = lambda: 1 << 20
        log = self.root / "run" / "11_eyes.log"
        self.assertEqual(worker.run(blend, ["add_eye_tracking.py"], log, []), engine.STOPPED)
        self.assertEqual(errors.classify("eyes", log.read_text())["code"], "VM-OOM")
        self.assertEqual(len(self.started), 2)

    def test_a_run_keeps_one_blender_for_its_steps_and_never_past_its_end(self):
        os.environ["OC2VRM_ONE_BLENDER"] = "1"
        project = self.made.project({"options": {}, "triangles": 40000})
        built = self.root / "01_build.blend"
        built.write_bytes(b"BLENDER-v402")
        project._record("build", {"status": "done", "blend": str(built)})

        def answer(order):
            Path(order["argv"][order["argv"].index("--") + 1]).write_bytes(b"BLENDER-v402")  # the step's own .blend
            return 0
        engine.subprocess.Popen = self.kept(answer)
        folder = self.root / "run"
        with project._kept_blender(folder):
            worker = project._worker
            entry = project._run_step("hair", engine.NAMES.index("hair"), folder)
            self.assertEqual((entry["status"], Path(entry["blend"]).name), ("done", "02_hair.blend"))
            self.assertNotIn("peak_mb", entry)  # one Blender for them all can't say what one step took
            self.assertEqual(len(self.told), 1)
            # the engine's switches for the step travel with it, the numbers beside its .blend among them
            self.assertEqual([kind for kind, _ in self.told[0]["run"]], ["expr", "expr", "file", "file"])
            self.assertEqual(self.told[0]["run"][2:], [["file", str(engine.FINGERPRINT)], ["file", str(engine.SCRIPTS / "fix_hair_weights.py")]])
            self.assertIsNotNone(worker.process)
        self.assertIsNone(project._worker)
        self.assertIsNone(worker.process)
        # switched off: no Blender is kept, and a step is run as it always was
        os.environ["OC2VRM_ONE_BLENDER"] = "0"
        with project._kept_blender(folder):
            self.assertIsNone(project._worker)

    def test_another_blender_starts_only_when_the_computer_has_room(self):
        os.environ.pop("OC2VRM_BLENDERS", None)
        project = self.made.project({"options": {}, "triangles": 40000})
        gb = 1 << 30
        engine.LOAD.idle_cores = lambda wait=0.0: 8.0
        engine.free_memory = lambda: 12 * gb
        self.assertTrue(project._room("check", {}, 0, calm=True))  # one always may
        self.assertEqual(project._need("check"), engine.BLENDER_NEEDS)
        self.assertTrue(project._room("check", {"export": 3 * gb}, 10 * gb))
        self.assertFalse(project._room("check", {"export": 3 * gb}, 10 * gb, calm=True))  # one has had to give way in this run
        self.assertFalse(project._room("check", {"export": 3 * gb, "springs": 3 * gb, "movement": 3 * gb}, 10 * gb))  # more than there was
        engine.free_memory = lambda: 4 * gb
        self.assertFalse(project._room("check", {"export": 3 * gb}, 10 * gb))  # it would leave under MEMORY_LOW
        # what the step took in this character's last build is gone by, with a margin
        project.state["steps"]["check"] = {"status": "stale", "peak_mb": 1024}
        self.assertEqual(project._need("check"), 1280 << 20)
        self.assertTrue(project._room("check", {"export": gb}, 10 * gb))
        engine.free_memory = lambda: None
        self.assertFalse(project._room("check", {"export": gb}, 10 * gb))  # a computer that doesn't say gets one at a time
        engine.free_memory = lambda: 12 * gb
        engine.LOAD.idle_cores = lambda wait=0.0: 1.2
        self.assertFalse(project._room("check", {"export": gb}, 10 * gb))  # the processor is busy with something else
        engine.LOAD.idle_cores = lambda wait=0.0: None
        self.assertTrue(project._room("check", {"export": gb}, 10 * gb))
        self.assertFalse(project._room("check", dict.fromkeys("abcd", gb), 100 * gb))  # never more than MOST_BLENDERS
        os.environ["OC2VRM_BLENDERS"] = "2"
        self.assertTrue(project._room("check", {"export": gb}, 10 * gb))
        self.assertFalse(project._room("check", {"export": gb, "springs": gb}, 10 * gb))
        os.environ["OC2VRM_BLENDERS"] = "1"
        self.assertFalse(project._room("check", {"export": gb}, 10 * gb))

    def test_setup_says_how_many_blenders_there_is_room_for_and_takes_the_users_number(self):
        os.environ.pop("OC2VRM_BLENDERS", None)
        gb = 1 << 30
        engine.LOAD.idle_cores = lambda wait=0.0: 10.0
        engine.free_memory = lambda: 32 * gb
        self.assertEqual(engine.blenders_now(), (4, ""))
        engine.free_memory = lambda: 9 * gb  # two at 3 GB each leave the 2 GB kept back, three don't
        self.assertEqual(engine.blenders_now(), (2, "9 GB of memory to be had"))
        engine.free_memory = lambda: 3 * gb
        self.assertEqual(engine.blenders_now()[0], 1)
        engine.free_memory = lambda: None
        self.assertEqual(engine.blenders_now()[0], 1)
        engine.free_memory = lambda: 32 * gb
        engine.LOAD.idle_cores = lambda wait=0.0: 4.4
        self.assertEqual(engine.blenders_now(), (2, f"4 of {os.cpu_count()} processor cores idle"))
        engine.LOAD.idle_cores = lambda wait=0.0: 10.0
        real, engine.setup = engine.setup, lambda given=None: {"set": True}
        try:
            self.assertEqual(engine.use_blenders("2"), {"set": True})
            self.assertEqual((engine.blenders_at_once(), engine.blenders_now()[0]), (2, 2))
            engine.use_blenders(1)
            self.assertEqual(engine.blenders_now(), (1, "set to one at a time"))
            engine.use_blenders("auto")
            self.assertIsNone(engine.blenders_at_once())
            with self.assertRaisesRegex(ValueError, "isn't a number of Blenders"):
                engine.use_blenders("many")
        finally:
            engine.setup = real
        speed = next(check for check in engine.setup()["checks"] if check["name"] == "Speed")
        self.assertRegex(speed["text"], r"^room for \d Blenders? at once as the computer stands")
        self.assertEqual(speed["state"], "ok")

    def side_by_side(self, project, takes, room=lambda name, running, budget, calm=False: True, free=None):
        """Run the last steps side by side with stand-ins for their Blenders. `takes[step]`: how long it
        runs (s), or what it raises. Returns (outcome, events, log): `log` holds (what, step, seconds since
        the start) for every start and end, in the order they happened."""
        import threading
        import time
        log, events, lock, began = [], [], threading.Lock(), time.time()

        def note(what, name):
            with lock:
                log.append((what, name, round(time.time() - began, 2)))

        def step(name, index, folder, kept=True, abandon=None, exported=None):
            note("start", name)
            self.assertFalse(kept)  # each beside the others in a Blender of its own
            if name == "vrm_check":
                self.exported = exported
            took = takes.get(name, 0.05)
            if isinstance(took, list):  # one for each time it is run
                took = took.pop(0)
            until = time.time() + (took if isinstance(took, (int, float)) else 0.05)
            while time.time() < until:
                if abandon.is_set():
                    note("aside", name)
                    raise engine.SetAside(name)
                time.sleep(0.01)
            note("end", name)
            if isinstance(took, BaseException):
                raise took
            return {"status": "done", **({"vrm": str(self.root / "14_export.vrm")} if name == "export" else {})}
        project._run_step = step
        if room is not None:
            project._room = room  # (else the engine's own: what the computer has, as the test has set it)
        project.portraits = lambda blend=None, on_event=None, locked=False: note("start", "portraits") or on_event({"step": "portraits", "status": "done"})
        if free is not None:
            engine.free_memory = free
        group = [*engine.TAIL, "portraits"]
        outcome = project._side_by_side(group, self.root / "run", events.append, newest=self.root / "12_mouth.blend")
        return outcome, events, log

    def most_at_once(self, log):
        going = most = 0
        for what, name, _ in log:
            going += {"start": 1, "end": -1, "aside": -1}[what] if name != "portraits" else 0
            most = max(most, going)
        return most

    def test_the_last_steps_run_side_by_side_and_the_files_check_waits_for_the_export(self):
        project = self.made.project({"options": {}, "triangles": 40000})
        outcome, events, log = self.side_by_side(project, {"export": 0.3, "check": 0.4, "springs": 0.4, "movement": 0.4, "vrm_check": 0.2})
        self.assertEqual({name: result["status"] for name, result in outcome.items() if name != "portraits"}, dict.fromkeys(engine.TAIL, "done"))
        self.assertIn("portraits", outcome)  # tried here: the chain doesn't draw them again
        # all that can start, at once, the export first; its check when it is over, and handed its file
        self.assertEqual([name for what, name, _ in log[:5]], ["export", "check", "springs", "movement", "portraits"])
        at = {(what, name): when for what, name, when in log}
        self.assertGreaterEqual(at["start", "vrm_check"], at["end", "export"])
        self.assertEqual(self.exported, self.root / "14_export.vrm")
        self.assertEqual(self.most_at_once(log), 4)
        self.assertLess(log[-1][2], 1.2)  # 0.3 + 0.2 on the longest line, not 1.7 one after another
        # each said as it starts, with the log the page shows the end of
        running = [(e["step"], Path(e["log"]).name) for e in events if e["status"] == "running"]
        self.assertEqual(running[:2] + [running[-1]], [("export", "14_export.log"), ("check", "13_check.log"), ("vrm_check", "15_vrm_check.log")])
        self.assertEqual([e for e in events if e["step"] == "portraits"], [{"step": "portraits", "status": "done"}])
        # ... and as it is over, ahead of its turn, for the page's bar: how it ended is the chain's to say
        ahead = [e["step"] for e in events if e["status"] == "ahead"]
        self.assertEqual((sorted(ahead), ahead[0], ahead[-1]), (sorted(engine.TAIL), "export", "vrm_check"))
        self.assertFalse(any(e["status"] in ("done", "failed") for e in events if e["step"] in engine.TAIL))

    def test_with_room_for_one_they_run_one_after_another(self):
        project = self.made.project({"options": {}, "triangles": 40000})
        outcome, _events, log = self.side_by_side(project, {}, room=lambda name, running, budget, calm=False: not running)
        self.assertEqual(len(outcome), 6)
        self.assertEqual([name for what, name, _ in log if what == "start"], ["export", "vrm_check", "check", "springs", "movement", "portraits"])
        self.assertEqual(self.most_at_once(log), 1)

    def test_what_ends_the_chain_ends_the_steps_after_it_and_no_others(self):
        project = self.made.project({"options": {}, "triangles": 40000})
        broken = RuntimeError("the export's own code tripped")
        outcome, _events, log = self.side_by_side(project, {"export": broken, "check": 0.3, "springs": 2.0, "movement": 2.0})
        self.assertIs(outcome["export"], broken)
        self.assertEqual(outcome["check"]["status"], "done")  # it comes before the export: it would have run
        self.assertEqual(set(outcome) - {"portraits"}, {"export", "check"})  # nothing after it is handed on
        self.assertNotIn(("start", "vrm_check"), [(what, name) for what, name, _ in log])
        self.assertEqual({name for what, name, _ in log if what == "aside"}, {"springs", "movement"})
        self.assertLess(log[-1][2], 1.5)  # ended, not waited for

    def test_a_step_that_fails_beside_others_is_run_again_alone_before_it_is_believed(self):
        # a graphics card or a drive that gave one Blender what it wanted may not give four
        os.environ.pop("OC2VRM_BLENDERS", None)
        engine.LOAD.idle_cores = lambda wait=0.0: 8.0  # (room for four: 16 GB to be had, each taken to need 3)
        project = self.made.project({"options": {}, "triangles": 40000})
        once = engine.StepFailed("check: failed with exit code 1")
        outcome, events, log = self.side_by_side(project, {"check": [once, 0.05], "export": 0.3, "springs": 0.3, "movement": 0.3}, room=None)
        self.assertEqual({name: result["status"] for name, result in outcome.items() if name != "portraits"}, dict.fromkeys(engine.TAIL, "done"))
        starts = [(name, when) for what, name, when in log if what == "start"]
        self.assertEqual([name for name, _ in starts].count("check"), 2)
        warned = [e["reason"] for e in events if e["status"] == "warning"]
        self.assertEqual(warned, ["check failed with other Blenders going beside it: it is run again by itself, and the rest of this build one at a time"])
        # the second time nothing else is going, and nothing starts beside another after
        going, calm = set(), False
        for what, name, _ in log:
            if name == "portraits":
                continue
            calm = calm or (what, name) == ("end", "check")
            if what == "start" and calm:
                self.assertEqual(going, set(), (name, log))
            going = going | {name} if what == "start" else going - {name}
        # one that fails alone as well is what it is: handed on, and with it what ends the chain there
        twice = engine.StepFailed("export: failed with exit code 1")
        outcome, _events, log = self.side_by_side(project, {"export": [twice, twice]}, room=None)
        self.assertIs(outcome["export"], twice)
        self.assertNotIn("vrm_check", outcome)
        self.assertNotIn(("start", "vrm_check"), [(what, name) for what, name, _ in log])
        # a check that only reports ends nothing: the others are handed on, and its failure with them
        told = engine.StepFailed("springs: failed with exit code 1")
        outcome, _events, _log = self.side_by_side(project, {"springs": [told, told]}, room=None)
        self.assertIs(outcome["springs"], told)
        self.assertEqual({name for name, result in outcome.items() if isinstance(result, dict)}, {"check", "export", "vrm_check", "movement"})

    def test_when_the_memory_runs_short_the_last_started_gives_way_and_runs_again_alone(self):
        import time
        os.environ.pop("OC2VRM_BLENDERS", None)
        engine.LOAD.idle_cores = lambda wait=0.0: 8.0
        project = self.made.project({"options": {}, "triangles": 40000})
        began = time.time()
        # room for four at the start (16 GB to be had, 3 GB taken as each one's need); short from 0.15 s on
        outcome, events, log = self.side_by_side(project, {"export": 0.5, "check": 0.5, "springs": 0.5, "movement": 0.5, "vrm_check": 0.1},
                                                 room=None, free=lambda: (16 << 30) if time.time() - began < 0.15 else (500 << 20))
        self.assertEqual({name: result["status"] for name, result in outcome.items() if name != "portraits"}, dict.fromkeys(engine.TAIL, "done"))
        aside = [name for what, name, _ in log if what == "aside"]
        self.assertEqual(aside, ["movement"])  # the one started last, once: after that nothing new starts beside another
        warned = [e for e in events if e["status"] == "warning"]
        self.assertEqual(len(warned), 1)
        self.assertIn("the memory ran short with 4 Blenders going at once: movement is run again by itself", warned[0]["reason"])
        starts = [name for what, name, _ in log if what == "start"]
        self.assertEqual(starts.count("movement"), 2)
        # from the moment it gave way, every start finds no other still going
        going, calm = set(), False
        for what, name, _ in log:
            if name == "portraits":
                continue
            if what == "aside":
                calm = True
            if what == "start" and calm:
                self.assertEqual(going, set(), (name, log))
            going = going | {name} if what == "start" else going - {name}

    def test_the_chain_takes_what_was_run_ahead_in_its_own_order(self):
        import time
        os.environ.pop("OC2VRM_BLENDERS", None)
        project = self.made.project({"options": {}, "triangles": 40000}, tpose="plain")
        project.data["face"] = None
        mouth = self.root / "09_tpose.blend"
        mouth.write_bytes(b"BLENDER-v402")
        project._record("build", {"status": "done", "blend": str(mouth)})
        engine.LOAD.idle_cores = lambda wait=0.0: 8.0
        events, order = [], []

        def step(name, index, folder, kept=True, abandon=None, exported=None):
            order.append((name, kept))
            time.sleep({"check": 0.3}.get(name, 0.05))
            if name == "movement":
                raise engine.StepFailed("movement: failed with exit code 1")
            return {"status": "done", "seconds": 1, **({"vrm": str(self.root / "14_export.vrm")} if name == "export" else {})}
        project._run_step = step
        project.portraits = lambda blend=None, on_event=None, locked=False: order.append(("portraits", None))
        real = engine._Lock, engine.provenance
        engine._Lock = lambda: __import__("contextlib").nullcontext()
        engine.provenance = lambda blender, manifest_path: {"blender_version": "5.2.2", "warnings": []}
        try:
            project.run(start="check", on_event=events.append)
        finally:
            engine._Lock, engine.provenance = real
        self.assertEqual({name for name, kept in order}, {"check", "export", "vrm_check", "movement", "springs", "portraits"})
        self.assertTrue(all(kept is False for name, kept in order if name != "portraits"))
        self.assertEqual([name for name, _ in order].count("portraits"), 1)
        # the state as one after another leaves it: each done in the build's order, the check that failed said so
        steps = project.state["steps"]
        self.assertEqual([steps[name]["status"] for name in engine.TAIL], ["done", "done", "done", "failed", "done"])
        ended = [(e["step"], e["status"]) for e in events if e["status"] in ("done", "failed") and e["step"] in engine.TAIL]
        self.assertEqual(ended, [("check", "done"), ("export", "done"), ("vrm_check", "done"), ("movement", "failed"), ("springs", "done")])
        # the check took longest and comes first in the build: the export's entry was not marked stale by it
        self.assertEqual(steps["export"]["status"], "done")
        # each said as it started: once, but for the check that failed in company and was run again by itself
        said = [e["step"] for e in events if e["status"] == "running" and e["step"] in engine.TAIL]
        self.assertEqual({name: said.count(name) for name in engine.TAIL}, {"check": 1, "export": 1, "vrm_check": 1, "movement": 2, "springs": 1})
        self.assertEqual([name for name, _ in order].count("movement"), 2)

    def test_a_blender_set_aside_says_so_and_is_no_failure(self):
        import threading
        test = self.made
        engine.subprocess.Popen, engine.time.sleep = test.hanging(), lambda seconds: None
        aside = threading.Event()
        aside.set()
        log = self.root / "aside.log"
        self.assertEqual(engine._blender("blender", None, ["check_springs.py"], log, [], abandon=aside), engine.ASIDE)
        self.assertTrue(test.killed)
        self.assertIn("SET ASIDE by the engine", log.read_text())
        self.assertIsNone(errors.classify("springs", log.read_text()))

    def test_what_a_steps_blender_took_is_kept_for_the_next_build(self):
        import subprocess
        self.assertIsNone(engine._peak_memory(SimpleNamespace(_handle=None)))  # a stand-in has no handle to ask of
        if os.name != "nt":
            self.skipTest("asked of Windows")
        took = subprocess.Popen([sys.executable, "-c", "x = bytearray(200 << 20)"])
        took.wait()
        self.assertTrue(200 << 20 < engine._peak_memory(took) < 600 << 20)
        self.assertTrue(0 <= engine.LOAD.idle_cores(wait=0.2) <= os.cpu_count())


class CharactersAtOnce(unittest.TestCase):
    """`engine.at_once`: the characters of one command (`regress`) built side by side, a Blender each,
    as many as `Project._room` finds room for."""

    def setUp(self):
        import threading
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.real = engine.LOCK, engine.free_memory, engine.LOAD, os.environ.pop("OC2VRM_BLENDERS", None)
        engine.LOCK = self.root / ".blender.lock"
        # a computer with 16 GB to be had and eight cores idle, not told how many Blenders it may run
        engine.free_memory = lambda: 16 << 30
        engine.LOAD = SimpleNamespace(idle_cores=lambda wait=0.0: 8.0)
        self.going, self.most, self.calls, self.count = 0, 0, [], threading.Lock()

    def tearDown(self):
        engine.LOCK, engine.free_memory, engine.LOAD, blenders = self.real
        if blenders is not None:
            os.environ["OC2VRM_BLENDERS"] = blenders
        engine.CANCEL.clear()
        self.folder.cleanup()

    def characters(self, count, gigabytes=5):
        """Stand-ins for characters whose rigging step takes `gigabytes`: asked for room by the engine's own rule."""
        import types
        made = []
        for index in range(count):
            character = SimpleNamespace(name=f"C{index}", _need=lambda name, need=gigabytes: need << 30)
            character._room = types.MethodType(Project._room, character)
            made.append(character)
        return made

    def work(self, fails=()):
        """A build that takes a moment; counts how many go at once. `fails`: (name, on which try) that end as a failed step."""
        import time

        def build(character, emit):
            with self.count:
                self.going += 1
                self.most = max(self.most, self.going)
                self.calls.append(character.name)
                attempt = self.calls.count(character.name)
            try:
                emit({"step": "build", "status": "running"})
                time.sleep(0.25)
                if (character.name, attempt) in fails:
                    raise engine.StepFailed(f"{character.name} failed in build")
                return f"{character.name} built"
            finally:
                with self.count:
                    self.going -= 1
        return build

    def test_as_many_at_once_as_there_is_room_for_and_each_in_its_place(self):
        # 16 GB to be had, 2 kept free, 5 a character: two fit, a third would pass what there was
        events = []
        got = engine.at_once(self.characters(5), self.work(), on_event=events.append, pause=0.02)
        self.assertEqual(got, [f"C{i} built" for i in range(5)])
        self.assertEqual(self.most, 2)
        # every event says whose it is
        self.assertEqual({e["character"] for e in events}, {f"C{i}" for i in range(5)})
        # the lock was held for them all and is let go; a build may take it again
        self.assertFalse(engine.LOCK.exists() or engine.SHARED.held)
        with engine._Lock():
            self.assertTrue(engine.LOCK.exists())
        # lighter characters, more of them, and never over the engine's own most
        self.most = 0
        engine.at_once(self.characters(8, gigabytes=1), self.work(), pause=0.02)
        self.assertEqual(self.most, engine.MOST_BLENDERS)

    def test_one_at_a_time_without_the_room_or_when_told_so(self):
        for scarce in ("memory", "cores", "unknown", "told"):
            self.most = 0
            real = engine.free_memory, engine.LOAD
            if scarce == "memory":
                engine.free_memory = lambda: 6 << 30   # a second character would leave under 2 GB
            elif scarce == "cores":
                engine.LOAD = SimpleNamespace(idle_cores=lambda wait=0.0: 1.0)
            elif scarce == "unknown":
                engine.free_memory = lambda: None      # a computer that doesn't say what it has left
            else:
                os.environ["OC2VRM_BLENDERS"] = "1"
            try:
                self.assertEqual(engine.at_once(self.characters(3), self.work(), pause=0.02), ["C0 built", "C1 built", "C2 built"], scarce)
                self.assertEqual(self.most, 1, scarce)
            finally:
                engine.free_memory, engine.LOAD = real
                os.environ.pop("OC2VRM_BLENDERS", None)

    def test_a_failure_beside_others_is_tried_again_alone_before_it_is_believed(self):
        events = []
        got = engine.at_once(self.characters(3), self.work(fails={("C1", 1)}), on_event=events.append, pause=0.02)
        self.assertEqual(got, ["C0 built", "C1 built", "C2 built"])
        self.assertEqual((self.calls.count("C1"), self.calls[-1]), (2, "C1"))  # again, last, by itself
        self.assertIn("built again by itself", next(e["reason"] for e in events if e["status"] == "warning"))
        # one that fails again alone has failed; so has one that failed alone the first time
        self.calls = []
        got = engine.at_once(self.characters(3), self.work(fails={("C1", 1), ("C1", 2)}), pause=0.02)
        self.assertIsInstance(got[1], engine.StepFailed)
        self.assertEqual((got[0], got[2], self.calls.count("C1")), ("C0 built", "C2 built", 2))
        self.calls = []
        got = engine.at_once(self.characters(1), self.work(fails={("C0", 1)}), pause=0.02)
        self.assertIsInstance(got[0], engine.StepFailed)
        self.assertEqual(self.calls, ["C0"])

    def test_each_run_takes_the_lock_inside_and_keeps_to_one_blender(self):
        seen = []

        def build(character, emit):
            with engine._Lock():  # as Project.run takes it: passed, since it is held for them all
                seen.append((engine.LOCK.exists(), engine.SHARED.held))
            seen.append(engine.LOCK.exists())  # ... and not let go by the one that passed
        engine.at_once(self.characters(2), build, pause=0.02)
        self.assertEqual(seen.count((True, True)), 2)
        self.assertEqual(seen.count(True), 2)
        # while others build or wait, a run's last steps are not run side by side: the room is counted
        # by characters. The last one left, and one asked for alone, asks for it as any build does
        made = Engine("test_survey_decides_what_was_left_to_it")
        made.setUp()
        try:
            project, asked = made.project({"options": {}, "triangles": 40000}), []
            project._room = lambda name, *args, **more: asked.append(name) or False
            ahead = lambda: project._ahead(engine.NAMES.index("check"), len(engine.NAMES) - 1, made.root, lambda event: None)  # noqa: E731
            with engine.together():
                engine.SHARED.crowd = lambda: 2
                self.assertEqual((ahead(), asked), ({}, []))
                engine.SHARED.crowd = lambda: 1
                self.assertEqual((ahead(), len(asked)), ({}, 1))
            self.assertEqual((engine.SHARED.held, engine.SHARED.crowd()), (False, 0))
        finally:
            made.tearDown()
        # outside it the lock is one build's alone, as before
        with engine._Lock():
            with self.assertRaisesRegex(engine.StepFailed, "another build is running"):
                with engine._Lock():
                    pass

    def test_stopped_by_the_user_nothing_more_is_started(self):
        def build(character, emit):
            engine.CANCEL.set()
            raise engine.Cancelled("stopped by the user in build")
        os.environ["OC2VRM_BLENDERS"] = "1"
        try:
            got = engine.at_once(self.characters(3), build, pause=0.02)
        finally:
            os.environ.pop("OC2VRM_BLENDERS", None)
        self.assertTrue(all(isinstance(end, engine.Cancelled) for end in got))
        self.assertFalse(engine.LOCK.exists())

    def test_a_regression_run_of_several_says_each_in_the_order_asked(self):
        import time
        made = Engine("test_survey_decides_what_was_left_to_it")
        made.setUp()
        real = Project.run, cli.REPO
        cli.REPO = made.root
        try:
            check = lambda head: {"poses": {"head turn": {"stretch_cm": head}}, "unweighted": 0, "worst": {}, "failed": []}  # noqa: E731
            paths = []
            for index, name in enumerate(("Anna", "Bea", "Cleo")):
                data = manifest.load(made.path)
                data["name"], data["out_dir"] = name, str(made.root / "out" / name)
                paths.append(made.root / f"{name}.json")
                manifest.save(data, paths[-1])
                project = Project(paths[-1], blender="blender")
                base = made.root / f"{name}_check.json"
                base.write_text(json.dumps(check(2.0)))
                project._record("check", {"status": "done", "json": str(base)})

            def run(scratch, on_event=None, **rest):
                with self.count:
                    self.going += 1
                    self.most = max(self.most, self.going)
                time.sleep(0.25)
                folder = scratch.out / "runs" / "20260101-100000"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "09_tpose.blend").write_text("x")
                (folder / "13_check.json").write_text(json.dumps(check(2.0 + len(scratch.name))))
                scratch._record("check", {"status": "done", "json": str(folder / "13_check.json")})
                with self.count:
                    self.going -= 1
            Project.run = run
            events = []
            result = cli.cmd_regress(SimpleNamespace(manifests=paths, keep=False, blender="blender", on_event=events.append))
            left = [sorted(p.name for p in (made.root / "out" / "regress" / name / "runs" / "20260101-100000").iterdir()) for name in ("Anna", "Bea", "Cleo")]
        finally:
            Project.run, cli.REPO = real
            made.tearDown()
        self.assertEqual([row["name"] for row in result["characters"]], ["Anna", "Bea", "Cleo"])
        self.assertEqual([row["worse"] for row in result["characters"]],
                         [[f"rig check, head turn: stretch 2.0 -> {2.0 + len(name):.1f} cm"] for name in ("Anna", "Bea", "Cleo")])
        self.assertEqual((result["worse"], result["failed"]), (3, []))
        self.assertGreater(self.most, 1)  # side by side, where there is the room
        # each one's models went as it ended, its reports stay
        self.assertEqual(left, [["13_check.json"]] * 3)
        self.assertEqual({e["character"] for e in events}, {"Anna", "Bea", "Cleo"})


class Failures(unittest.TestCase):
    """Every failure and refusal handed to the page drawn: a code, what happened, what to do and the
    option that does it (app/errors.py `failure`), or no code and the log's telling line for one the app
    has no name for. The page reads none of a log's words: it once did, and told a tester who had marked
    the face to mark it."""

    def setUp(self):
        self.made = Engine("test_survey_decides_what_was_left_to_it")
        self.made.setUp()
        self.root, self.path = self.made.root, self.made.path
        self.real = engine._blender, engine.LOCK
        engine.LOCK = self.root / ".blender.lock"

    def tearDown(self):
        engine._blender, engine.LOCK = self.real
        self.made.tearDown()

    def project(self, **options):
        return self.made.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}}, **options)

    def blender(self, said, code=1, makes=False):
        """Every Blender of a run ends with `code`, having written `said` (a dict by script name, or
        one text for all) into its log; with `makes`, also the file the step is asked for."""
        def blender(blender, blend_in, args, log, extra):
            script = Path(str(args[0])).stem
            text = said.get(script, "") if isinstance(said, dict) else said
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text(text)
            return code if (text or not isinstance(said, dict)) else 0
        engine._blender = blender

    def test_the_eye_steps_refusals_are_told_apart_by_the_engine(self):
        # one code, as before (the fallback goes by it), and what happened said for each
        log = "wrote 11_eyes.blend\nFAILED: face over the eye patch, left eye looking Out with eyeBlinkRight: 136 of 1550 points\nFAILED: face over the eye patch, right eye: 9\n"
        patch = errors.classify("eyes", log)
        self.assertEqual(patch["code"], "VM-EYES-REFUSED")
        self.assertIn("would go under the face", patch["what"])
        self.assertEqual(patch["detail"], "face over the eye patch, left eye looking Out with eyeBlinkRight: 136 of 1550 points")
        # the advice never names the way that was just refused
        self.assertIsNone(patch["options"])
        self.assertNotIn("slide", patch["fix"])
        iris = errors.classify("eyes", "FAILED: iris shows over the lids, right eye looking Down with eyeBlinkRight: 17 of 379 points\n")
        self.assertEqual((iris["code"], iris["options"]), ("VM-EYES-REFUSED", {"eyes": "slide"}))
        self.assertIn("blink doesn't cover the moving iris", iris["what"])
        discs = errors.classify("eyes", "FAILED: the eyes aren't modelled as iris discs narrower than the eyes: nothing to slide\n")
        self.assertEqual(discs["options"], {"eyes": None})
        # a refusal in words the table doesn't have keeps the code's own, with what was said
        other = errors.classify("eyes", "FAILED: left iris over the lids\n")
        self.assertEqual((other["what"], other["detail"]), (errors.describe("VM-EYES-REFUSED")["what"], "left iris over the lids"))

    def test_a_refused_patch_gets_the_engines_advice_whatever_the_table_says(self):
        # (the table gives the patch no option: the engine knows whether the model has eye pieces to slide instead)
        log = "FAILED: face over the eye patch, left eye looking In: 80 of 1550 points\n"
        refused = engine.StepFailed("eyes: failed", errors.classify("eyes", log))
        advice = self.project(eyes="slide")._eyes_advice("eyes", refused)
        self.assertEqual(advice.known["options"], {"eyes": "modelled"})
        self.assertIn("would go under the face", advice.known["what"])

    def test_a_failure_is_kept_as_the_page_draws_it(self):
        project = self.project(eyes="slide")
        project._record("mouth", {"status": "done", "blend": str(self.root / "x.blend")})
        (self.root / "x.blend").write_bytes(b"BLENDER")
        self.blender("wrote 11_eyes.blend\nFAILED: face over the eye patch, left eye looking In: 80 of 1550 points\nBlender quit\n")
        project.state["steps"]["face_tracking"] = {"status": "done", "blend": str(self.root / "x.blend")}
        project.run(only="eyes")
        kept = project.state["steps"]["eyes"]
        self.assertEqual(kept["status"], "failed")
        drawn = kept["failure"]
        self.assertEqual((drawn["code"], drawn["page"], drawn["options"]), ("VM-EYES-REFUSED", "face", {"eyes": "modelled"}))
        self.assertIn("would go under the face", drawn["what"])
        self.assertIn("left eye looking In", drawn["detail"])
        self.assertIn("set eyes to modelled", drawn["fix"])
        # ... and as it always was, for whoever reads the state by its old fields
        self.assertEqual(kept["known"]["code"], "VM-EYES-REFUSED")
        self.assertIn("FAILED: face over the eye patch", kept["reason"])
        shown = next(s for s in Project(self.path, blender="blender").status()["steps"] if s["step"] == "eyes")
        self.assertEqual(shown["failure"], drawn)

    def test_a_failure_with_no_name_says_the_logs_telling_line_and_no_code(self):
        project = self.project()
        self.blender("Read blend\nTraceback (most recent call last):\n  File \"fix_fingers.py\", line 3\n"
                     "ValueError: found 2 digits standing out from the hand, need 5\n\nError: script failed, file: 'fix_fingers.py', exiting.\n"
                     "Blender 5.2.2 LTS (hash d13f752e3b9c built 2026-09-15)\n\nBlender quit\n")
        project.state["steps"]["pieces"] = {"status": "done", "blend": str(self.root / "x.blend")}
        (self.root / "x.blend").write_bytes(b"BLENDER")
        project.run(only="finger_guides")
        drawn = project.state["steps"]["finger_guides"]["failure"]
        self.assertEqual((drawn["code"], drawn["options"]), (None, None))
        self.assertEqual(drawn["detail"], "ValueError: found 2 digits standing out from the hand, need 5")
        self.assertIn("no name for", drawn["what"])
        self.assertIn("report", drawn["fix"])
        self.assertNotIn("known", project.state["steps"]["finger_guides"])

    def test_the_telling_line_of_a_log_is_the_error_not_what_blender_printed_after_it(self):
        # a tester's build of v1, as its state kept it: the error, then the FBX importer's and the baker's lines
        log = ("FBX version: 7400\nTraceback (most recent call last):\n  File \"make_vrm.py\", line 107, in <module>\n    main()\n    ~~~~^^\n"
               "  File \"face_points.py\", line 270, in _patch_mapping\n    raise FacePointsError(\"the face patch has no polygons\")\n"
               "vtuber_maker.pipeline.face_points.FacePointsError: the face patch has no polygons\nFBX version: 7400\n"
               "Info: Baking map saved to internal image, save it externally or pack it\n\nError: script failed, file: 'make_vrm.py', exiting.\n"
               "Blender 5.2.2 LTS (hash d13f752e3b9c built 2026-09-15 01:37:04)\n\nBlender quit\n")
        self.assertEqual(errors.last_line(log), "FacePointsError: the face patch has no polygons")
        self.assertEqual(errors.failure("build", None, log)["detail"], "FacePointsError: the face patch has no polygons")
        # a script that ends with a sentence of its own (no traceback): the sentence, not Blender's talk after it
        self.assertEqual(errors.last_line("Read blend\nno hand had fingers to rig; nothing written\nInfo: Saved copy\n\nBlender quit\n"),
                         "no hand had fingers to rig; nothing written")
        # of two tracebacks the last is the one that ended it
        self.assertEqual(errors.last_line("Traceback (most recent call last):\n  File \"a.py\"\nKeyError: 'x'\n\nDuring handling of the above "
                                          "exception, another exception occurred:\n\nTraceback (most recent call last):\n  File \"b.py\"\n"
                                          "ValueError: no bone Head in the rig\n"), "ValueError: no bone Head in the rig")
        self.assertEqual(errors.last_line(""), "")

    def test_a_step_that_stops_the_chain_is_kept_drawn_beside_its_last_good_output(self):
        project = self.project()
        self.blender("LandmarkError: the arms don't stand clear of the body: they hang at its sides\n")
        with self.assertRaises(engine.StepFailed) as failed:
            project.run(only="build")
        self.assertEqual(failed.exception.step, "build")
        drawn = project.state["steps"]["build"]["last_failure"]["failure"]
        self.assertEqual((drawn["code"], drawn["page"]), ("VM-ARMS-AT-SIDES", "model"))
        self.assertIn("T-pose", drawn["fix"])

    def test_a_state_from_before_failures_were_kept_drawn_reads_the_same(self):
        # as v14 and every version before it wrote them: the reason's end, and `known` when there was a code
        old = {"steps": {
            "survey": {"status": "done", "json": str(self.root / "survey.json")},
            "eyes": {"status": "failed", "finished": "2026-10-10T01:22:00", "reason": "wrote 11_eyes.blend\nFAILED: face over the eye patch, right eye looking Up: 16 of 1452 points\n",
                     "known": {"code": "VM-EYES-REFUSED", "what": errors.describe("VM-EYES-REFUSED")["what"],
                               "fix": errors.describe("VM-EYES-REFUSED")["fix"], "options": {"eyes": "slide"}}},
            "finger_guides": {"status": "failed", "finished": "2026-10-11T22:00:00",
                              "reason": "ValueError: found 2 digits standing out from the hand, need 5\n\nError: script failed, file: 'x', exiting.\nBlender quit\n"},
            "build": {"status": "done", "blend": "good.blend", "last_failure": {
                "finished": "2026-10-09T10:00:00", "reason": "build: Blender ran out of memory (see x.log)",
                "known": {"code": "VM-OOM", "what": "Blender ran out of memory", "fix": "close other programs"}}}}}
        self.project()
        (self.root / "out").mkdir(exist_ok=True)
        (self.root / "out" / "state.json").write_text(json.dumps(old))
        steps = {s["step"]: s for s in Project(self.path, blender="blender").status()["steps"]}
        eyes = steps["eyes"]["failure"]
        self.assertIn("would go under the face", eyes["what"])
        self.assertEqual(eyes["detail"], "face over the eye patch, right eye looking Up: 16 of 1452 points")
        # (the old advice named the way that had just been refused: not said again)
        self.assertEqual((eyes["options"], "slide" in eyes["fix"]), (None, False))
        self.assertEqual(steps["finger_guides"]["failure"]["detail"], "ValueError: found 2 digits standing out from the hand, need 5")
        self.assertIsNone(steps["finger_guides"]["failure"]["code"])
        self.assertEqual((steps["build"]["last_failure"]["failure"]["code"], steps["build"]["last_failure"]["failure"]["fix"]),
                         ("VM-OOM", "close other programs"))
        # nothing of the file is written over by being read
        self.assertEqual(json.loads((self.root / "out" / "state.json").read_text()), old)

    def test_an_optional_repair_that_crashes_is_a_failure_and_the_build_goes_on_without_it(self):
        # the loose step's script falling over read as "nothing to do": the avatar built quietly without the repair
        project = self.project()
        (self.root / "x.blend").write_bytes(b"BLENDER")
        project.state["steps"]["face_tracking"] = {"status": "done", "blend": str(self.root / "x.blend")}
        self.blender("Traceback (most recent call last):\n  File \"fix_loose_pieces.py\", line 80\nKeyError: 'Head'\n\nError: script failed, file: 'x', exiting.\n")
        project.run(only="loose")  # (not raised: the chain goes on with the file before it)
        kept = project.state["steps"]["loose"]
        self.assertEqual((kept["status"], kept["failure"]["code"], kept["failure"]["detail"]), ("failed", None, "KeyError: 'Head'"))
        self.assertEqual(project.latest(), self.root / "x.blend")
        # one that says it had nothing to do is believed, as before
        self.blender("no skinned mesh in the file\n")
        project.run(only="loose")
        self.assertEqual(project.state["steps"]["loose"]["status"], "nothing to do")
        # and one the engine itself stopped had something to do: the chain ends there
        self.blender("Malloc returns null\n", code=engine.STOPPED)
        with self.assertRaises(engine.StepFailed) as stopped:
            project.run(only="loose")
        self.assertEqual(stopped.exception.known["code"], "VM-OOM")

    def test_what_the_engine_refuses_has_a_code(self):
        project = self.project()
        with self.assertRaises(engine.StepFailed) as nothing:
            project.run(only="tpose")
        self.assertEqual(nothing.exception.known["code"], "VM-NOTHING-BUILT")
        self.assertEqual(project.state["steps"]["tpose"]["last_failure"]["failure"]["page"], "build")
        with self.assertRaises(engine.StepFailed) as no_file:
            project.send_to_folder()
        self.assertEqual(no_file.exception.known["code"], "VM-NO-EXPORT")
        with self.assertRaises(engine.StepFailed) as early:
            project.approve()
        self.assertEqual(early.exception.known["code"], "VM-NO-EXPORT")
        self.assertEqual(engine.NoFolder("no folder").known["code"], "VM-NO-FOLDER")
        self.assertTrue(issubclass(engine.NoFolder, FileNotFoundError))
        # a model not surveyed: every later step reads the survey
        project.state["steps"].pop("survey")
        with self.assertRaises(engine.StepFailed) as unsurveyed:
            project.run(only="build")
        self.assertEqual(unsurveyed.exception.known["code"], "VM-NO-SURVEY")
        # another build holding the lock
        engine.LOCK.write_text(f"pid {os.getpid()} since now")
        try:
            with self.assertRaisesRegex(engine.StepFailed, "another build is running") as busy:
                project.run(only="survey")
        finally:
            engine.LOCK.unlink()
        self.assertEqual((busy.exception.known["code"], busy.exception.known["detail"]), ("VM-BUSY", f"pid {os.getpid()} since now"))
        # a record of the builds that can't be read, nor its backup
        project._save_state()
        project.state_path.write_text("{ half a state")
        project.state_path.with_name("state.json.bak").unlink(missing_ok=True)
        with self.assertRaises(engine.StepFailed) as unread:
            Project(self.path, blender="blender")
        self.assertEqual(unread.exception.known["code"], "VM-STATE-UNREADABLE")
        self.assertEqual(server.busy("build").known["code"], "VM-BUSY")
        # every code a refusal names is one the app can say
        for code in ("VM-BUSY", "VM-NO-SURVEY", "VM-NOTHING-BUILT", "VM-NO-EXPORT", "VM-STATE-UNREADABLE", "VM-FOLDER-FILE", "VM-NO-FOLDER"):
            self.assertTrue(errors.describe(code)["what"] and errors.describe(code)["fix"], code)

    def test_a_job_that_fails_hands_the_page_its_failure_drawn(self):
        import time
        jobs = server.Jobs()

        def ends_at_build(emit):
            error = engine.StepFailed("build: failed with exit code 1 (see x.log)\nKeyError: 'Hips'\nBlender quit", None)
            error.step = "build"
            raise error
        jobs.start("build", ends_at_build)
        while jobs.running:
            time.sleep(0.01)
        failure = jobs.poll()["failure"]
        self.assertEqual((failure["step"], failure["code"], failure["detail"]), ("build", None, "KeyError: 'Hips'"))
        # a refusal with a code
        jobs.start("build", lambda emit: (_ for _ in ()).throw(engine.refused("VM-NO-SURVEY", "run the survey step first")))
        while jobs.running:
            time.sleep(0.01)
        self.assertEqual((jobs.poll()["failure"]["code"], jobs.poll()["error"]), ("VM-NO-SURVEY", "run the survey step first"))
        # a sentence of the app's own (bad input) is said as it is, and the next job starts clean
        jobs.start("init", lambda emit: (_ for _ in ()).throw(ValueError("the name can't have ? in it")))
        while jobs.running:
            time.sleep(0.01)
        self.assertEqual((jobs.poll()["failure"], jobs.poll()["error"]), (None, "the name can't have ? in it"))

    def test_the_page_reads_no_word_of_a_log(self):
        # what the Build page says of a failure comes from `failure`: no pattern over a reason is left in it
        core = (server.GUI / "js" / "core.js").read_text(encoding="utf-8")
        build = (server.GUI / "js" / "build.js").read_text(encoding="utf-8")
        jobs = (server.GUI / "js" / "jobs.js").read_text(encoding="utf-8")
        pages = core + build + jobs
        for words in ("plain(", "lastLine", "iris shows over the lids", "face over the eye patch", "no folder to send", ".match(/see", "reason).split"):
            self.assertFalse(words in pages, f"the page still reads a failure's words: {words}")
        self.assertTrue("tell(s.failure)" in build and "state.failure" in jobs)


class Preflight(unittest.TestCase):
    """What is known to go wrong with a build before a Blender starts (app/preflight.py): each line fires
    on what had the problem and says nothing of what is right. The limits behind the lines were held
    against the characters on the developers' computer (SETUP.md "Preflight: what is known to go wrong
    before a build"); these hold what each line reads and when it is said."""

    # a face marked as model A's is, in the 2048 px picture of the face patch
    FACE = {"image": "face_patch.png", "size": [2048, 2048],
            "right_eye": {"inner": [911, 1418], "outer": [715, 1403], "top": [800, 1373], "bottom": [814, 1512]},
            "left_eye": {"inner": [1131, 1416], "outer": [1332, 1402], "top": [1235, 1367], "bottom": [1235, 1508]},
            "mouth": {"left": [958, 1677], "right": [1069, 1684], "top": [1013, 1649], "bottom": [1013, 1710]}}
    SURVEY = {"options": {}, "triangles": 52000, "texture": [4096, 4096], "face_patch": True,
              "pose": {"kind": "T-pose", "arm_drop_degrees": 3.3}, "parts": {"read": True, "eye_inserts": True},
              "orientation": {"up_sure": True, "facing_sure": True, "turned": False}, "notes": []}

    def setUp(self):
        self.made = Engine("test_survey_decides_what_was_left_to_it")
        self.made.setUp()
        self.root, self.path = self.made.root, self.made.path
        (self.root / "model.glb").write_bytes(b"the model")
        self.mark(self.FACE)
        # a computer with a Blender and its add-on, whatever this one has
        self.real = engine.blender_version, engine.vrm_addon_version, engine.find_blender
        engine.blender_version, engine.vrm_addon_version = lambda blender: "5.2.2", lambda blender, version: "4.7.2"
        # ... and its system drive not looked at, but where a test says so (the tests' folder may be on another);
        # nor OneDrive, and a browser found (class TestersComputer says what is said of those)
        self.system = os.environ.pop("SystemDrive", None)
        self.onedrive = {name: os.environ.pop(name) for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial") if name in os.environ}
        from app import sheet
        self.find_browser, sheet.find_browser = sheet.find_browser, lambda given=None: "msedge.exe"

    def tearDown(self):
        engine.blender_version, engine.vrm_addon_version, engine.find_blender = self.real
        if self.system is not None:
            os.environ["SystemDrive"] = self.system
        os.environ.update(self.onedrive)
        from app import sheet
        sheet.find_browser = self.find_browser
        self.made.tearDown()

    def mark(self, points):
        (self.root / "face.json").write_text(json.dumps(points))

    def project(self, survey=None, **options):
        return self.made.project({**self.SURVEY, **(survey or {})}, **options)

    def found(self, project, **steps):
        """{code: line} of what a look at `project` stops at or warns of."""
        said = project.preflight(**steps)
        self.assertEqual(said["skipped"], [])
        return {line["code"]: line for line in said["lines"] if line["kind"] != "ok"}

    def pieces(self, *pieces):
        data = manifest.load(self.path)
        data["pieces"] = [{"name": name, "point": point, "count": count, "role": role} for name, point, count, role in pieces]
        manifest.save(data, self.path)

    def listing(self, rows, **more):
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        (out / "pieces.json").write_text(json.dumps({"model": str(self.root / "model.glb"), "pieces": [
            {"label": i, "count": count, "point": point, "suggested": suggested, "why": why}
            for i, (point, count, suggested, why) in enumerate(rows)], **more}))

    def built(self, project, **report):
        """`project` as it stands built to its end today, its rigging step having reported `report`."""
        project.state["steps"]["build"] = {"status": "done", "report": report}
        project.state["steps"]["vrm_check"] = {"status": "done", "finished": "2099-01-01T00:00:00"}
        return project

    def test_a_character_with_nothing_wrong_is_not_stopped_or_warned(self):
        said = self.project().preflight()
        self.assertEqual((said["stops"], said["warns"], said["skipped"]), (0, 0, []))
        # ... and an agent is told what was looked at
        self.assertEqual([line["code"] for line in said["lines"]], ["OK-TOOLS", "OK-MODEL", "OK-FACE", "OK-ROOM"])
        self.assertIn("Blender 5.2.2 with the VRM Add-on 4.7.2", said["lines"][0]["text"])
        self.assertEqual((said["from"], said["to"]), ("survey", "springs"))

    def test_every_line_has_a_code_the_app_knows_and_says_what_to_do(self):
        import re
        source = (REPO / "app" / "preflight.py").read_text(encoding="utf-8")
        codes = set(re.findall(r'"(VM-[A-Z-]+)"', source))
        self.assertGreater(len(codes), 20)
        for code in codes:
            known = errors.describe(code)
            self.assertIsNotNone(known, code)
            self.assertTrue(known["what"] and known["fix"], code)
        # what the computer ran short of is never promised to fail again
        self.assertFalse(any(errors.repeats(code) for code in ("VM-OOM", "VM-DISK-FULL", "VM-TIMEOUT", "VM-BAD-OUTPUT", "VM-NO-SUCH-CODE")))
        self.assertTrue(errors.repeats("VM-ARMS-AT-SIDES") and errors.repeats("VM-EYES-REFUSED"))

    def test_its_limits_are_the_pipelines_own(self):
        # survey.py and list_pieces.py import bpy: their numbers and words are held here by reading them
        import re
        survey = (REPO / "addon" / "vtuber_maker" / "pipeline" / "survey.py").read_text(encoding="utf-8")
        from app import preflight
        said = {name: int(value.replace("_", "")) for name, value in re.findall(r"^(HEAVY_TRIANGLES|COARSE_TRIANGLES) = ([\d_]+)", survey, re.M)}
        self.assertEqual(said, {"HEAVY_TRIANGLES": preflight.HEAVY_TRIANGLES, "COARSE_TRIANGLES": preflight.COARSE_TRIANGLES})
        listing = (REPO / "scripts" / "list_pieces.py").read_text(encoding="utf-8")
        self.assertIn(f'"{preflight.THE_FACE}"', listing)
        self.assertIn(f'"{preflight.AN_INSERT}', listing)

    def test_no_blender_an_old_one_or_no_add_on_stops_any_run(self):
        project = self.project()
        engine.blender_version = lambda blender: None
        self.assertEqual(self.found(project)["VM-NO-BLENDER"]["kind"], "stop")
        engine.blender_version = lambda blender: "4.1.1"
        self.assertEqual(self.found(project)["VM-BLENDER-OLD"]["page"], "setup")
        engine.blender_version, engine.vrm_addon_version = lambda blender: "5.2.2", lambda blender, version: None
        # ... of a run of one check alone too: nothing runs without them
        self.assertEqual(list(self.found(project, only="springs")), ["VM-NO-VRM-ADDON"])

        def none(given=None):
            raise FileNotFoundError("Blender 4.2 or later not found")
        engine.find_blender = none
        self.assertEqual(self.found(Project(self.path))["VM-NO-BLENDER"]["kind"], "stop")

    def test_a_file_the_character_names_and_is_gone(self):
        data = manifest.load(self.path)
        data["rig"], data["paint"] = "rig.fbx", "paint/overlays"
        manifest.save(data, self.path)
        gone = [line for line in self.project().preflight()["lines"] if line["kind"] != "ok"]
        self.assertEqual([(line["kind"], line["code"]) for line in gone], [("stop", "VM-FILE-GONE"), ("warn", "VM-FILE-GONE")])
        self.assertIn("rig.fbx", gone[0]["text"])
        (self.root / "model.glb").unlink()
        self.assertEqual(self.found(self.project())["VM-NO-MODEL"]["kind"], "stop")
        self.assertIn("model.glb", self.found(self.project())["VM-NO-MODEL"]["text"])

    def rigged_glb(self, value):
        """A GLB with one skin whose inverse bind matrix is all `value`."""
        body = json.dumps({"skins": [{"inverseBindMatrices": 0, "joints": [0]}], "nodes": [{}],
                           "accessors": [{"bufferView": 0, "count": 1, "componentType": 5126, "type": "MAT4"}],
                           "bufferViews": [{"buffer": 0, "byteLength": 64}], "buffers": [{"byteLength": 64}]}).encode()
        body += b" " * (-len(body) % 4)
        binary = struct.pack("<16f", *[value] * 16)
        (self.root / "rig.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(body) + len(binary)) + struct.pack("<I4s", len(body), b"JSON")
                                           + body + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
        data = manifest.load(self.path)
        data["rig"] = "rig.glb"
        manifest.save(data, self.path)

    def test_a_rigged_glb_whose_bones_would_all_import_broken(self):
        # Tripo's rigged GLB came with bind matrices like 1e31: every bone at NaN (model E's)
        self.rigged_glb(1e31)
        self.assertEqual(self.found(self.project())["VM-RIG-GLB"]["kind"], "stop")
        self.rigged_glb(1.0)
        self.assertEqual(self.found(self.project()), {})

    def test_a_colour_multiplied_by_a_grey_factor_is_seen_and_read_through(self):
        # newer Tripo exports: a Mix node once imported, which stopped the texture steps (VM-TRIPO-MIX) until
        # every step read through it. Said as looked at, never as a fault: models that carry it build
        def model(factor):
            body = json.dumps({"materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, **factor}}]}).encode()
            body += b" " * (-len(body) % 4)
            (self.root / "model.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(body)) + struct.pack("<I4s", len(body), b"JSON") + body)
            said = self.project().preflight()
            self.assertEqual((said["stops"], said["warns"]), (0, 0))
            return [line["text"] for line in said["lines"] if line["code"] == "OK-COLOUR-FACTOR"]
        self.assertEqual(model({}), [])
        self.assertEqual(model({"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}), [])
        self.assertIn("multiplied by 0.8", model({"baseColorFactor": [0.8, 0.8, 0.8, 1.0]})[0])
        # a step that did fail on it is promised to fail again, as any failure its inputs decide
        self.assertTrue(errors.repeats("VM-TRIPO-MIX"))

    def test_what_the_survey_measured_of_the_model(self):
        for survey, code, kind in (({"texture": None}, "VM-NO-TEXTURE", "stop"),
                                   ({"pose": {"kind": "arms down", "arm_drop_degrees": 84.0}}, "VM-ARMS-DOWN", "stop"),
                                   ({"triangles": 1_050_000}, "VM-HEAVY-MODEL", "stop"),
                                   ({"triangles": 2_400}, "VM-COARSE-MODEL", "warn"),
                                   # model A's stand-in decimated in Blender across its texture seams: 1,740 pieces of 163
                                   ({"pieces": 1_740}, "VM-FRAGMENTS", "stop"),
                                   ({"parts": {"read": False}}, "VM-PARTS-NOT-READ", "warn")):
            said = self.found(self.project(survey))
            self.assertEqual([(c, line["kind"], line["page"]) for c, line in said.items()], [(code, kind, "model")], code)
        self.assertIn("84", self.found(self.project({"pose": {"kind": "arms down", "arm_drop_degrees": 84.0}}))["VM-ARMS-DOWN"]["text"])
        # an A-pose builds (models A and B are 73 and 61 degrees down), and so does a note of joints placed
        # by proportion, which the survey makes of half the good characters
        fine = {"pose": {"kind": "A-pose", "arm_drop_degrees": 73.4}, "notes": ["the shoulders were placed from typical proportions"],
                "pieces": 239}  # (model B's count, the most of any character here)
        self.assertEqual(self.found(self.project(fine)), {})
        # none of it is said of a run that doesn't rig
        self.assertEqual(self.found(self.project({"texture": None}), start="tpose"), {})

    def test_an_orientation_in_doubt_is_asked_until_the_character_has_been_built_and_seen(self):
        unsure = {"orientation": {"up_sure": False, "facing_sure": True, "turned": False}}
        project = self.project(unsure)
        project.state["steps"]["survey"]["finished"] = "2026-10-12T10:00:00"
        self.assertEqual(self.found(project)["VM-ORIENT-UNSURE"]["kind"], "stop")
        self.assertIn("which way up", self.found(project)["VM-ORIENT-UNSURE"]["text"])
        project.state["steps"]["vrm_check"] = {"status": "done", "finished": "2026-10-12T10:05:00"}
        self.assertEqual(self.found(project)["VM-ORIENT-UNSURE"]["kind"], "warn")
        # surveyed again since (another model): in doubt again
        project.state["steps"]["survey"]["finished"] = "2026-10-12T11:00:00"
        self.assertEqual(self.found(project)["VM-ORIENT-UNSURE"]["kind"], "stop")

    def test_face_points_that_cannot_be_a_face(self):
        import copy

        def moved(**parts):
            points = copy.deepcopy(self.FACE)
            for part, change in parts.items():
                for name, (dx, dy) in change.items():
                    for key in ([name] if name != "all" else list(points[part])):
                        points[part][key] = [points[part][key][0] + dx, points[part][key][1] + dy]
            return points

        def said(points):
            self.mark(points)
            line = self.found(self.project()).get("VM-FACE-MARKS")
            return line and (line["kind"], line["page"], line["text"])
        # model D's fault, on a face whose eyes are marked: the mouth saved again 300 px down the picture
        kind, page, text = said(moved(mouth={"all": (0, 300)}))
        self.assertEqual((kind, page), ("stop", "face"))
        self.assertIn("on the chin or the throat", text)
        self.assertIn("off to one side", said(moved(mouth={"all": (260, 0)}))[2])
        self.assertIn("at the height of the eyes", said(moved(mouth={"all": (0, -260)}))[2])
        self.assertIn("far above the other", said(moved(left_eye={"all": (0, -200)}))[2])
        self.assertIn("on the same eye", said(moved(left_eye={"all": (-400, 0)}))[2])
        self.assertIn("outside the picture", said(moved(mouth={"bottom": (0, 500)}))[2])
        self.assertIn("corners are marked on the same spot", said(moved(right_eye={"inner": (-196, -15)}))[2])
        self.assertIn("mouth's two corners", said(moved(mouth={"left": (111, 7)}))[2])
        # what is right: the marks as they are; a mouth 200 px lower (a long face: the pirate's is 0.91 eye
        # distances down, this 1.0); a chibi's, close under the eyes; the two eyes kept under each other's
        # names (model B's are, and she builds right); the mouth alone, the eyes being covered
        swapped = dict(self.FACE, right_eye=self.FACE["left_eye"], left_eye=self.FACE["right_eye"])
        for right in (self.FACE, moved(mouth={"all": (0, 200)}), moved(mouth={"all": (0, -80)}), swapped,
                      {key: value for key, value in self.FACE.items() if not key.endswith("_eye")}):
            self.assertIsNone(said(right))
        # points that can't be read at all are another line, and the only one
        self.mark({key: value for key, value in self.FACE.items() if key != "mouth"})
        self.assertEqual(list(self.found(self.project())), ["VM-FACE-FILE"])
        (self.root / "face.json").write_text("{ half a file")
        self.assertIn("isn't JSON", self.found(self.project())["VM-FACE-FILE"]["text"])
        # ... and no points at all is worth knowing, not a fault
        data = manifest.load(self.path)
        data["face"] = None
        manifest.save(data, self.path)
        self.assertEqual([(code, line["kind"]) for code, line in self.found(self.project()).items()], [("VM-NO-FACE-POINTS", "warn")])

    def test_marks_made_on_another_framing_of_the_face_patch(self):
        # the points are pixels in the picture the patch was made from: a patch made again framed otherwise
        # (a demon girl's, first framed up among her horns) leaves them off the face
        face = self.root / "out" / "face"
        face.mkdir(parents=True)
        (face / "head_at.json").write_text(json.dumps({"head_at": 0.923, "body": 1.0}))
        project = self.project()
        self.assertEqual(self.found(project), {})  # marks from before the frame was kept say nothing
        cli.save_face(self.path, self.FACE, blender="blender")
        kept = json.loads((self.root / "out" / "face.json").read_text())
        self.assertEqual(kept["frame"], {"head_at": 0.923, "body": 1.0})
        self.assertEqual(self.found(self.project()), {})
        (face / "head_at.json").write_text(json.dumps({"head_at": 0.901, "body": 1.0}))
        line = self.found(self.project())["VM-FACE-REFRAMED"]
        self.assertEqual((line["kind"], line["page"]), ("stop", "face"))
        # (a millimetre's difference in how a survey rounds is the same frame)
        (face / "head_at.json").write_text(json.dumps({"head_at": 0.924, "body": 1.0}))
        self.assertEqual(self.found(self.project()), {})

    HEAD = ([0.0, -0.1, 1.4], 1898, "body", "the face"), ([0.02, -0.11, 1.41], 243, "body", "an eye or mouth insert (face tracking moves it)")
    HAIR = ([0.0, 0.0, 1.5], 3385, "hair", "reads as hair"), ([0.0, -0.1, 1.5], 1000, "hair", "reads as hair"), ([0.1, 0.0, 1.4], 168, "hair", "reads as hair")

    def test_a_swinging_role_on_the_face_and_hair_named_a_hat(self):
        # a tester's fox girl, every suggestion of a listing made before her eyes were marked accepted: her
        # face called hair, four of her hair pieces hats. Her face swung on hair springs and her hair stood still
        self.listing([*self.HEAD, *self.HAIR], eyes_marked=True)
        self.pieces(("hair_7792", self.HEAD[0][0], 1898, "hair"), ("hat_8351", self.HEAD[1][0], 243, "hat"),
                    ("hat_6783", self.HAIR[0][0], 3385, "hat"), ("hat_6546", self.HAIR[1][0], 1000, "hat"), ("hair_10329", self.HAIR[2][0], 168, "hair"))
        said = self.found(self.project())
        self.assertEqual([(code, line["kind"], line["page"], line["action"]) for code, line in said.items()],
                         [("VM-ROLE-ON-FACE", "stop", "parts", "remove_pieces"), ("VM-HAIR-AS-HAT", "stop", "parts", "remove_pieces")])
        self.assertEqual(said["VM-ROLE-ON-FACE"]["args"], {"names": ["hair_7792"]})
        self.assertIn("the face the eyes are marked on", said["VM-ROLE-ON-FACE"]["text"])
        self.assertEqual(said["VM-HAIR-AS-HAT"]["args"], {"names": ["hat_6783", "hat_6546"]})
        self.assertIn("4,385 of its 4,553 vertices", said["VM-HAIR-AS-HAT"]["text"])
        # an eye piece on a swinging role is the same fault; as a hat it only follows the head
        self.pieces(("cloth", self.HEAD[1][0], 243, "swing_cloth"))
        self.assertIn("an eye or mouth piece", self.found(self.project())["VM-ROLE-ON-FACE"]["text"])
        # what is right: the hair called hair, one small bit of it a hat, a hat on what is no hair
        self.pieces(("hair_a", self.HAIR[0][0], 3385, "hair"), ("clip", self.HAIR[2][0], 168, "hat"), ("eye", self.HEAD[1][0], 243, "hat"))
        self.assertEqual(self.found(self.project()), {})
        # ... and all of it a hat when the user asked for a solid head themselves
        self.pieces(("hat_6783", self.HAIR[0][0], 3385, "hat"), ("hat_6546", self.HAIR[1][0], 1000, "hat"))
        self.assertIn("VM-HAIR-AS-HAT", self.found(self.project()))
        self.assertEqual(self.found(self.project(rigid_head=True)), {})

    def test_roles_given_from_a_listing_made_before_the_eyes_were_marked(self):
        self.listing([*self.HEAD, *self.HAIR])  # no word of the eyes: a listing of before v13, or before the marks
        self.pieces(("hat_6783", self.HAIR[0][0], 3385, "hat"), ("hair_7792", self.HEAD[0][0], 1898, "hair"))
        line = self.found(self.project())["VM-LISTING-STALE"]
        self.assertEqual((line["kind"], line["page"], line["action"]), ("warn", "parts", "list_pieces"))
        # built to its end with these marks, hair swinging: its user has seen the face and the hair move
        self.assertEqual(self.found(self.built(self.project(), hair_chains=12)), {})
        # ... but hats on a head no hair swings from are still worth the look
        self.assertIn("VM-LISTING-STALE", self.found(self.built(self.project(), hair_chains=0)))
        # roles that have nothing to do with the head say nothing, listed when they may
        self.listing([*self.HEAD, *self.HAIR, ([0.0, 0.0, 0.8], 900, "body", "nothing special")])
        self.pieces(("skirt", [0.0, 0.0, 0.8], 900, "fitted_skirt"))
        self.assertEqual(self.found(self.project()), {})
        # and a character whose eyes aren't marked has no face to hold a role against
        self.pieces(("hat_6783", self.HAIR[0][0], 3385, "hat"))
        self.mark({key: value for key, value in self.FACE.items() if not key.endswith("_eye")})
        self.assertEqual(self.found(self.project()), {})

    def test_a_piece_named_that_the_model_no_longer_has(self):
        # the build refuses a piece whose vertex count isn't the model's (roles.find: "a different or re-exported model?")
        self.listing([*self.HEAD, *self.HAIR], eyes_marked=True)
        self.pieces(("skirt", [0.0, 0.0, 0.8], 777, "fitted_skirt"))
        line = self.found(self.project())["VM-PIECE-GONE"]
        self.assertEqual((line["kind"], line["args"]), ("stop", {"names": ["skirt"]}))
        # a listing of another model, or one older than the model's file, says nothing of this one
        listed = json.loads((self.root / "out" / "pieces.json").read_text())
        (self.root / "out" / "pieces.json").write_text(json.dumps({**listed, "model": str(self.root / "other.glb")}))
        self.assertEqual(self.found(self.project()), {})
        (self.root / "out" / "pieces.json").write_text(json.dumps(listed))
        os.utime(self.root / "model.glb", (4102444800, 4102444800))
        self.assertEqual(self.found(self.project()), {})

    def test_eye_tracking_chosen_and_no_eyes_marked(self):
        self.mark({key: value for key, value in self.FACE.items() if not key.endswith("_eye")})
        self.assertEqual([(code, line["kind"]) for code, line in self.found(self.project(eyes="iris")).items()], [("VM-EYES-UNMARKED", "warn")])
        self.assertEqual(self.found(self.project(eyes=None)), {})

    def test_a_fallback_remembered_from_before_the_face_points_were_saved_again(self):
        # "a character keeps building sliding eyes": the patch an earlier build fell back to is remembered
        project = self.project()
        project.state["eyes_fallback"] = "slide"
        project.state["steps"]["eyes"] = {"status": "done", "finished": "2020-01-01T00:00:00"}
        line = self.found(project)["VM-EYES-FALLBACK"]
        self.assertEqual((line["kind"], line["action"]), ("warn", "forget_eyes_fallback"))
        # the page's button: forgotten, and the eye step is to be built again
        self.assertEqual(project.forget_eyes_fallback(), {"forgot": "slide"})
        self.assertEqual((project.state.get("eyes_fallback"), project.state["steps"]["eyes"]["status"]), (None, "stale"))
        self.assertEqual(self.found(project), {})
        # marks older than the eye step, a way of the model's own discs, or eyes chosen by the user: nothing to say
        for fell, chosen, finished in (("slide", None, "2099-01-01T00:00:00"), ("modelled", None, "2020-01-01T00:00:00"),
                                       ("slide", "slide", "2020-01-01T00:00:00")):
            project = self.project(eyes=chosen)
            project.state["eyes_fallback"] = fell
            project.state["steps"]["eyes"] = {"status": "done", "finished": finished}
            self.assertEqual(self.found(project), {}, (fell, chosen, finished))

    def failed(self, project, step, code, soft=True, **known):
        """`step` as a run that started from the manifest as it is now left it: failed with `code`."""
        from app import preflight
        import hashlib
        run = project.out / "runs" / "20261012-120000"
        run.mkdir(parents=True, exist_ok=True)
        (run / "run.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()[:16],
                                                  "commit": preflight._commit()}))
        log = run / f"11_{step}.log"
        log.write_text("FAILED: face over the eye patch")
        failure = {"reason": "...", "finished": "2099-01-01T00:00:00", "known": {**errors.describe(code), **known}, "log": str(log)}
        project.state["steps"][step] = {"status": "failed", **failure} if soft else {"status": "not run", "last_failure": failure}
        return project

    def test_what_failed_last_time_and_would_fail_the_same_way_again(self):
        # a tester pressed Build on an eye mode the model can't take, got the same refusal each time,
        # and sent a report headed "face points done, still failing"
        project = self.failed(self.project(eyes="slide"), "eyes", "VM-EYES-REFUSED", fix="set eyes to modelled", options={"eyes": "modelled"})
        line = self.found(project)["VM-EYES-REFUSED"]
        self.assertEqual((line["kind"], line["page"], line["options"], line["fix"]), ("stop", "face", {"eyes": "modelled"}, "set eyes to modelled"))
        self.assertIn('"eyes" failed this way on the last build', line["text"])
        # of a run that goes through the eye step, and no other
        self.assertEqual(self.found(project, only="check"), {})
        self.assertIn("VM-EYES-REFUSED", self.found(project, start="tpose"))
        # anything it reads changed since: not promised (the choice, here)
        cli.cmd_set(SimpleNamespace(manifest=self.path, key="eyes", value='"modelled"'))
        again = Project(self.path, blender="blender")
        again.state = project.state
        self.assertEqual(self.found(again), {})
        # left on automatic, the build tries the other ways by itself ...
        project = self.failed(self.project(eyes=None), "eyes", "VM-EYES-REFUSED")
        self.assertEqual(self.found(project), {})
        # ... painted eyes too, which slide from the start: the blink alone is left to try
        project = self.failed(self.project({"parts": {"read": True, "eye_inserts": False}}, eyes=None), "eyes", "VM-EYES-REFUSED")
        self.assertEqual(self.found(project), {})
        # ... and once the blink alone failed too there is nothing else
        project.state["eyes_fallback"] = "blink"
        self.assertIn("VM-EYES-REFUSED", self.found(project))
        # a step that stopped the chain, the same: arms the survey couldn't find stay unfound
        project = self.failed(self.project(), "build", "VM-ARMS-AT-SIDES", soft=False)
        self.assertEqual(self.found(project)["VM-ARMS-AT-SIDES"]["page"], "model")
        # what the computer ran short of may well work the next time, and a failure with no name is not known
        for code in ("VM-OOM", "VM-DISK-FULL", "VM-TIMEOUT", "VM-BAD-OUTPUT"):
            self.assertEqual(self.found(self.failed(self.project(), "build", code, soft=False)), {}, code)
        # ... nor is one from a run that kept no record of what it was started from
        project = self.failed(self.project(), "build", "VM-ARMS-AT-SIDES", soft=False)
        (project.out / "runs" / "20261012-120000" / "run.json").unlink()
        self.assertEqual(self.found(project), {})

    def test_files_saved_by_a_newer_blender_than_the_one_chosen_now(self):
        # a file saved by 5.2 can't be opened by 4.2
        project = self.project()
        run = project.out / "runs" / "20261012-120000"
        run.mkdir(parents=True)
        (run / "02_hair.blend").write_bytes(b"BLENDER")
        (run / "run.json").write_text(json.dumps({"blender_version": "5.2.2"}))
        project.state["steps"]["hair"] = {"status": "done", "blend": str(run / "02_hair.blend")}
        self.assertEqual(self.found(project, start="tpose"), {})
        engine.blender_version = lambda blender: "4.2.1"
        line = self.found(project, start="tpose")["VM-BLEND-NEWER"]
        self.assertEqual((line["kind"], line["page"]), ("stop", "setup"))
        self.assertIn("5.2.2", line["text"])
        self.assertEqual(self.found(project), {})  # from the start nothing of it is opened

    def test_too_little_room_on_the_drive_or_in_memory(self):
        project = self.project()
        engine.free_disk = lambda path: 600 << 20
        self.assertEqual(self.found(project)["VM-DISK-FULL"]["kind"], "stop")
        # room for a step, not for the build: it would be stopped part way
        engine.free_disk = lambda path: 1500 << 20
        self.assertEqual(self.found(project)["VM-DISK-LOW"]["kind"], "stop")
        # ... which a run of one check alone doesn't write: worth knowing there, no more
        self.assertEqual(self.found(project, only="springs")["VM-DISK-LOW"]["kind"], "warn")
        engine.free_disk = lambda path: 2500 << 20
        self.assertEqual(self.found(project)["VM-DISK-LOW"]["kind"], "warn")
        engine.free_disk = lambda path: 100 << 30
        # memory: against what this character's heaviest step took the last time
        project.state["steps"]["build"] = {"status": "done", "peak_mb": 2900}
        engine.free_memory = lambda: 2500 << 20
        line = self.found(project)["VM-MEMORY-LOW"]
        self.assertEqual(line["kind"], "stop")
        self.assertIn("took 2.8 GB", line["text"])
        self.assertEqual(self.found(project, only="check"), {})  # a check takes a third of that
        engine.free_memory = lambda: 3500 << 20
        self.assertEqual(self.found(project), {})
        # a character never built: what a rigging step is taken to need
        engine.free_memory = lambda: 1900 << 20
        self.assertIn("is taken to need", self.found(self.project())["VM-MEMORY-LOW"]["text"])
        # a system that doesn't say what it has left is not held to anything
        engine.free_memory = lambda: None
        self.assertEqual(self.found(self.project()), {})
        # Windows' page file grows on the system drive: said when that one is nearly full and is another
        # than the build's (this computer's C:, at 3 GB, while its builds go to E:)
        if os.name == "nt":
            other = next(letter for letter in "CD" if not str(self.root.resolve()).upper().startswith(letter))
            os.environ["SystemDrive"] = f"{other}:"
            engine.free_disk = lambda path: (3 << 30) if str(path).upper().startswith(other) else (100 << 30)
            line = self.found(self.project())["VM-SYSTEM-DISK-LOW"]
            self.assertEqual((line["kind"], line["text"][:18]), ("warn", f"{other}: has 3.0 GB free"))
            del os.environ["SystemDrive"]

    def test_a_stop_built_past_is_asked_of_that_character_once(self):
        from app import preflight
        project = self.project({"pose": {"kind": "arms down", "arm_drop_degrees": 84.0}})
        self.assertEqual(preflight.accept(project, ["VM-ARMS-DOWN", "VM-NO-TEXTURE"]), {"accepted": ["VM-ARMS-DOWN"]})
        line = self.found(project)["VM-ARMS-DOWN"]
        self.assertEqual((line["kind"], line["accepted"]), ("warn", True))
        self.assertEqual(json.loads((project.out / "state.json").read_text())["accepted"]["VM-ARMS-DOWN"], line["text"])
        # what it says has changed (another model, lower arms): asked again
        project.state["steps"]["survey"]["json"] = str(self.root / "other.json")
        (self.root / "other.json").write_text(json.dumps({**self.SURVEY, "pose": {"kind": "arms down", "arm_drop_degrees": 89.0}}))
        self.assertEqual(self.found(project)["VM-ARMS-DOWN"]["kind"], "stop")

    def test_a_look_that_cannot_be_made_never_stands_in_the_way(self):
        project = self.project()
        project.state["steps"]["survey"]["json"] = str(self.root / "survey.json")
        (self.root / "survey.json").write_text("{ half a survey")
        said = project.preflight()
        self.assertEqual((said["lines"], said["stops"]), ([], 0))
        self.assertIn("everything", said["skipped"][0])
        # ... and one part that can't be made leaves the others said
        project = self.project({"pose": "not what a survey says"})
        said = project.preflight()
        self.assertEqual([part.split(":")[0] for part in said["skipped"]], ["model"])
        self.assertIn("OK-ROOM", [line["code"] for line in said["lines"]])

    def test_a_finished_vrm_is_asked_for_its_licence_and_nothing_of_a_model(self):
        # a VRM 0.x avatar whose licence forbids changed versions: the user's to read and say, never set for them
        vrm = self.root / "avatar.vrm"
        body = json.dumps({"extensions": {"VRM": {"meta": {"title": "A", "licenseName": "CC_BY_ND"}, "humanoid": {"humanBones": []}}}}).encode()
        body += b" " * (-len(body) % 4)
        vrm.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(body)) + struct.pack("<I4s", len(body), b"JSON") + body)
        data = manifest.load(self.path)
        data["source"], data["face"] = "avatar.vrm", None
        manifest.save(data, self.path)
        line = self.found(Project(self.path, blender="blender"))["VM-LICENCE"]
        self.assertEqual((line["kind"], line["page"], line["options"]), ("stop", "options", None))
        self.assertIn("doesn't allow changed versions", line["text"])
        cli.cmd_set(SimpleNamespace(manifest=self.path, key="licence_read", value="true"))
        self.assertEqual(self.found(Project(self.path, blender="blender")), {})

    def test_the_command_says_it_and_a_strict_run_does_not_start(self):
        import contextlib
        import io
        from unittest import mock
        self.project({"pose": {"kind": "arms down", "arm_drop_degrees": 84.0}})._save_state()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--blender", "blender", "preflight", str(self.path)]), 0)
        said = json.loads(out.getvalue())
        self.assertEqual((said["stops"], said["lines"][0]["code"], said["lines"][0]["page"]), (1, "VM-ARMS-DOWN", "model"))
        with mock.patch.object(Project, "run", return_value={"ran": True}) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as events:
                self.assertEqual(cli.main(["--blender", "blender", "run", str(self.path), "--strict"]), 1)
            self.assertIn("VM-ARMS-DOWN", json.loads(out.getvalue())["error"])
            run.assert_not_called()
            # without the word the run starts, the stop said first among its events
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as events:
                self.assertEqual(cli.main(["--blender", "blender", "run", str(self.path)]), 0)
            run.assert_called_once()
        first = json.loads(events.getvalue().splitlines()[0])
        self.assertEqual((first["status"], first["kind"], first["code"]), ("preflight", "stop", "VM-ARMS-DOWN"))

    def test_the_page_reads_it_with_a_get_and_remembers_with_a_post(self):
        self.project({"pose": {"kind": "arms down", "arm_drop_degrees": 84.0}})._save_state()
        self.assertIn("preflight", server.READ_ONLY)
        self.assertFalse({"preflight_accept", "forget_eyes_fallback"} & server.READ_ONLY)
        real = server.Project
        server.Project = lambda path, blender=None: real(path, blender="blender")
        try:
            said = server.api("preflight", {"manifest": [str(self.path)], "from": ["tpose"]}, {})
            self.assertEqual((said["from"], said["stops"]), ("tpose", 0))
            self.assertEqual(server.api("preflight", {"manifest": [str(self.path)]}, {})["stops"], 1)
            with self.assertRaises(ValueError):
                server.api("preflight", {"manifest": [str(self.path)], "from": ["../x"]}, {})
            self.assertEqual(server.api("preflight_accept", {}, {"manifest": str(self.path), "codes": ["VM-ARMS-DOWN"]}), {"accepted": ["VM-ARMS-DOWN"]})
            self.assertEqual(server.api("preflight", {"manifest": [str(self.path)]}, {})["stops"], 0)
        finally:
            server.Project = real


class TestersComputer(unittest.TestCase):
    """What a tester's computer does that the developers' doesn't: folders with spaces and letters of other
    alphabets, deep ones, OneDrive, a build killed in its middle, a drive that fills, no Edge or Chrome."""

    def setUp(self):
        self.made = Engine("test_survey_decides_what_was_left_to_it")
        self.made.setUp()
        self.root, self.path = self.made.root, self.made.path
        self.real = engine._blender, engine.LOCK, engine.blender_version, engine.vrm_addon_version
        engine.LOCK = self.root / ".blender.lock"
        engine.blender_version, engine.vrm_addon_version = lambda blender: "5.2.2", lambda blender, version: "4.7.2"
        self.environment = dict(os.environ)
        for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial", "SystemDrive"):
            os.environ.pop(name, None)

    def tearDown(self):
        engine._blender, engine.LOCK, engine.blender_version, engine.vrm_addon_version = self.real
        os.environ.clear()
        os.environ.update(self.environment)
        self.made.tearDown()

    def project(self, **options):
        (self.root / "model.glb").write_bytes(b"the model")
        (self.root / "face.json").write_text(json.dumps(Preflight.FACE))
        return self.made.project(Preflight.SURVEY, **options)

    def warned(self, project, **steps):
        return {line["code"]: line for line in project.preflight(**steps)["lines"] if line["kind"] != "ok"}

    def test_a_blender_ends_with_the_app_that_started_it(self):
        # the app's window closed in the middle of a build: its Blender ran on for minutes, holding gigabytes,
        # beside the next build, which had rightly taken the lock a dead process left
        import subprocess
        import time
        if os.name != "nt":
            self.skipTest("a Windows job; elsewhere a closed terminal takes its children along")
        code = ("import subprocess, sys, time\nsys.path.insert(0, {repo!r})\nfrom app import engine\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                "tied = engine.dies_with_us(child) if {tie} else None\nprint(child.pid, tied, flush=True)\ntime.sleep(120)\n")
        ended = {}
        for tie in (True, False):
            app = subprocess.Popen([sys.executable, "-c", code.format(repo=str(REPO), tie=tie)], stdout=subprocess.PIPE, text=True)
            pid, tied = app.stdout.readline().split()
            self.assertEqual(tied, str(tie) if tie else "None")
            app.kill()  # as Task Manager ends it: no chance to tidy up
            app.wait()
            app.stdout.close()
            for _ in range(50):
                if not engine._alive(int(pid)):
                    break
                time.sleep(0.1)
            ended[tie] = not engine._alive(int(pid))
            if not ended[tie]:
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        # tied, it went with the app; left alone (as before), it outlived it
        self.assertEqual(ended, {True: True, False: False})
        # a stand-in for a process (the tests') is not tied, and nothing is raised for it
        self.assertFalse(engine.dies_with_us(SimpleNamespace(_handle=None)))

    def test_a_step_stopped_on_a_full_drive_gives_back_what_it_had_half_written(self):
        project = self.project()
        (self.root / "x.blend").write_bytes(b"BLENDER")
        project.state["steps"]["hair"] = {"status": "done", "blend": str(self.root / "x.blend")}
        seen = {}

        def blender(blender, blend_in, args, log, extra):
            # half a save, Blender's own "@" of it, and the drive is full: the engine stops it
            stem = Path(args[1]).with_suffix("")
            for kind in (".blend", ".blend@"):
                stem.with_suffix(kind).write_bytes(b"BLENDER half")
            seen["stem"] = stem
            Path(log).write_text("Saving...\n\nSTOPPED by the engine after 12 s: disk full\n")
            engine.free_disk = lambda path: 10 << 20
            return engine.STOPPED
        engine._blender = blender
        with self.assertRaises(engine.StepFailed) as stopped:
            project.run(only="tpose")
        self.assertEqual(stopped.exception.known["code"], "VM-DISK-FULL")
        self.assertEqual([p.name for p in seen["stem"].parent.glob("*.blend*")], [])
        self.assertEqual(project.state["steps"]["tpose"]["last_failure"]["failure"]["page"], "home")
        # the file the step read is not the step's to remove
        self.assertTrue((self.root / "x.blend").exists())

    def test_the_apps_own_write_refused_by_a_full_drive_is_said_as_that(self):
        import contextlib
        import errno
        import io
        import time
        from unittest import mock
        full = OSError(errno.ENOSPC, "No space left on device")
        self.assertTrue(engine.no_space(full))
        windows = OSError(22, "There is not enough space on the disk")
        windows.winerror = 112
        self.assertTrue(engine.no_space(windows))
        self.assertFalse(engine.no_space(OSError(errno.EACCES, "denied")) or engine.no_space(ValueError("x")))
        jobs = server.Jobs()
        with mock.patch.object(server, "_log_error"):
            jobs.start("build", lambda emit: (_ for _ in ()).throw(full))
            while jobs.running:
                time.sleep(0.01)
        self.assertEqual((jobs.poll()["failure"]["code"], jobs.poll()["failure"]["page"]), ("VM-DISK-FULL", "home"))
        # on the command line too: a sentence and its code, not a traceback
        self.project()._save_state()
        with mock.patch.object(Project, "status", side_effect=full), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--blender", "blender", "status", str(self.path)]), 1)
        self.assertEqual(json.loads(out.getvalue())["code"], "VM-DISK-FULL")
        # any other trouble of the system's is not taken for it
        with mock.patch.object(Project, "status", side_effect=OSError(errno.EACCES, "denied")):
            with self.assertRaises(OSError):
                cli.main(["--blender", "blender", "status", str(self.path)])

    def test_a_record_of_builds_that_cannot_be_read_is_set_aside_at_the_users_word(self):
        project = self.project()
        project._record("build", {"status": "done", "blend": "a.blend"})
        with self.assertRaisesRegex(ValueError, "nothing to set aside"):  # one that reads is left alone
            engine.set_state_aside(self.path)
        project.state_path.write_text('{"steps": {"bui', encoding="utf-8")
        project.state_path.with_name("state.json.bak").write_text("", encoding="utf-8")
        with self.assertRaises(engine.StepFailed) as unread:
            Project(self.path, blender="blender")
        self.assertEqual(unread.exception.known["code"], "VM-STATE-UNREADABLE")
        moved = engine.set_state_aside(self.path)["aside"]
        self.assertEqual(len(moved), 2)
        self.assertEqual(Path(moved[0]).read_text(encoding="utf-8"), '{"steps": {"bui')  # kept as it was, for a report
        self.assertEqual(Project(self.path, blender="blender").state, {"steps": {}})
        self.assertNotIn("state_aside", server.READ_ONLY)

    def test_folders_with_spaces_and_letters_of_other_alphabets(self):
        # a tester's Documents is "C:\\Users\\José María\\Documents", a model "キャラ 1.glb"
        import contextlib
        import io
        home = self.root / "José María" / "Mes modèles ñ 日本語"
        home.mkdir(parents=True)
        model = home / "キャラ 1.glb"
        model.write_bytes(b"glTF-not-really")
        manifest_path = home / "out put" / "character.json"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--blender", "blender", "init", str(model), "--name", "Zoë", "--manifest", str(manifest_path)]), 0)
        made = json.loads(out.getvalue())
        self.assertEqual(Path(made["manifest"]), manifest_path.resolve())
        project = Project(manifest_path, blender="blender")
        self.assertEqual(manifest.path_of(project.data, "source"), model.resolve())
        # the manifest is UTF-8 whatever the computer's own code page, and reads back the same
        self.assertEqual(json.loads(manifest_path.read_bytes().decode("utf-8"))["name"], "Zoë")
        # a relative path to it is the same character
        here = os.getcwd()
        os.chdir(home)
        try:
            self.assertEqual(Project(Path("out put") / "character.json", blender="blender").manifest_path, manifest_path.resolve())
        finally:
            os.chdir(here)
        # what a Blender is started with: every path whole, one argument each, and in the step's log as it is
        command = engine._command("blender", home / "a b.blend", ["x.py", model, home / "o ut.vrm"], [])
        self.assertIn(str(model), command)
        self.assertIn(str(home / "a b.blend"), command)
        started = {}

        class Ran:
            returncode = 0

            def __init__(self, command, **kwargs):
                started["command"] = command

            def poll(self):
                return 0
        real = engine.subprocess.Popen
        engine.subprocess.Popen = Ran
        try:
            log = home / "logs ñ" / "01 build.log"
            self.assertEqual(engine._blender("blender", None, ["x.py", model], log, []), 0)
        finally:
            engine.subprocess.Popen = real
        self.assertIn(str(model), started["command"])
        self.assertIn("キャラ 1.glb", log.read_text(encoding="utf-8"))
        # ... and what the run's one Blender is told, which crosses a pipe as JSON
        order = json.loads(json.dumps({"blend": str(home / "a b.blend"), "log": str(log)}))
        self.assertEqual(order["blend"], str(home / "a b.blend"))
        # the state and its backup, written and read there
        project._record("survey", {"status": "done", "json": str(home / "survey ñ.json")})
        self.assertEqual(Project(manifest_path, blender="blender").state["steps"]["survey"]["json"], str(home / "survey ñ.json"))

    def test_a_folder_too_deep_for_windows_is_said_before_the_build(self):
        if os.name != "nt":
            self.skipTest("Windows' own limit")
        deep = self.root / ("a" * 60) / ("b" * 60) / ("c" * 60)
        data = manifest.load(self.path)
        data["out_dir"] = str(deep)
        manifest.save(data, self.path)
        line = self.warned(self.project())["VM-PATH-LONG"]
        self.assertEqual(line["kind"], "stop")
        self.assertIn(str(len(str(deep.resolve()))), line["text"])
        # one that leaves room for the longest file a build writes says nothing
        data["out_dir"] = str(self.root / "out")
        manifest.save(data, self.path)
        self.assertNotIn("VM-PATH-LONG", self.warned(self.project()))
        from app import preflight
        self.assertLess(len(str((self.root / "out").resolve())) + preflight.LONGEST_WRITTEN, preflight.WINDOWS_PATH)
        # the longest is a close-up of the rig check's, by the names scripts/check_avatar.py gives them
        # (one character short of it, a folder at the limit built until the check drew a stretch)
        script = (REPO / "scripts" / "check_avatar.py").read_text(encoding="utf-8")
        words = [("stretch", "squash"), ("front", "back", "side"), ("rest", "posed")]
        for word in (w for group in words for w in group):
            self.assertIn(f'"{word}"', script)
        name = max((f"{kind}_{view}_{state}.png" for kind in words[0] for view in words[1] for state in words[2]), key=len)
        self.assertEqual(preflight.LONGEST_WRITTEN, len(f"\\runs\\20261012-155109\\{engine.NAMES.index('check'):02d}_check_spots\\{name}"))

    def test_builds_written_inside_onedrive_and_a_model_kept_online_only(self):
        from unittest import mock
        from app import preflight
        project = self.project()
        self.assertEqual(self.warned(project), {})
        os.environ["OneDriveCommercial"] = str(self.root)
        line = self.warned(project)["VM-ONEDRIVE"]
        self.assertEqual(line["kind"], "warn")
        self.assertIn(str(self.root), line["text"])
        os.environ["OneDriveCommercial"] = str(self.root / "elsewhere")
        self.assertEqual(self.warned(project), {})
        with mock.patch.object(preflight, "_online_only", return_value=True):
            self.assertEqual(self.warned(project)["VM-MODEL-ONLINE"]["page"], "model")
        self.assertFalse(preflight._online_only(self.root / "model.glb"))  # a file that is simply there

    def test_no_edge_or_chrome_is_said_and_holds_nothing_up(self):
        from unittest import mock
        from app import sheet
        project = self.project()
        with mock.patch.object(sheet, "find_browser", side_effect=sheet.NoBrowser("no Edge or Chrome found")):
            line = self.warned(project)["VM-NO-BROWSER"]
            self.assertEqual((line["kind"], line["at"]), ("warn", "vrm_check"))
            # only of a run that comes to the exported file's check
            self.assertEqual(self.warned(project, start="survey", stop="tpose"), {})
            self.assertEqual(project.preflight()["stops"], 0)

    def test_no_drive_with_room_for_the_browser_is_said_before_a_build(self):
        from unittest import mock
        from app import sheet
        project = self.project()
        with mock.patch.object(sheet, "find_browser", return_value="msedge.exe"):
            with mock.patch.object(sheet, "free_at", return_value=4 << 30):
                line = self.warned(project)["VM-NOT-DRAWN"]
                self.assertEqual((line["kind"], line["at"]), ("warn", "vrm_check"))
                self.assertIn("4.0 GB", line["text"])
                self.assertEqual(self.warned(project, start="survey", stop="tpose"), {})
                self.assertEqual(project.preflight()["stops"], 0)
            with mock.patch.object(sheet, "free_at", return_value=6 << 30):
                self.assertNotIn("VM-NOT-DRAWN", self.warned(project))

    def test_no_blender_found_is_a_refusal_with_its_code(self):
        from unittest import mock
        with mock.patch.object(engine, "installed_blenders", return_value=[]), mock.patch.object(engine.shutil, "which", return_value=None), \
                mock.patch.dict(os.environ, {"BLENDER": ""}):
            with self.assertRaises(FileNotFoundError) as none:
                engine.find_blender()
        self.assertEqual(none.exception.known["code"], "VM-NO-BLENDER")


class Inputs(unittest.TestCase):
    """`python -m app inputs` (app/inputs.py): a model's transformed copies built and held against the model's
    own build. What the copies are is scripts/make_variants.py's, in Blender; what is judged of them is here."""

    def setUp(self):
        self.made = Engine("test_survey_decides_what_was_left_to_it")
        self.made.setUp()
        self.root, self.path = self.made.root, self.made.path

    def tearDown(self):
        self.made.tearDown()

    def check(self, name, stretch):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"failed": [], "unweighted": 0, "poses": {"arms raised": {"stretch_cm": stretch, "squash": 0.8}}}))
        return path

    def test_a_copy_is_held_to_the_models_own_build_or_to_the_stop_it_should_meet(self):
        from app import inputs
        same = {"outcome": "same", "codes": []}
        self.assertTrue(inputs._as_expected(same, "same"))
        # stopped where the same was hoped for keeps the rule (it was said) and is still not what was expected
        self.assertFalse(inputs._as_expected({"outcome": "stopped", "codes": ["VM-ORIENT-UNSURE"]}, "same"))
        self.assertFalse(inputs._as_expected({"outcome": "worse", "codes": []}, "same"))
        self.assertTrue(inputs._as_expected({"outcome": "stopped", "codes": ["VM-ARMS-AT-SIDES"]}, "VM-ARMS-DOWN VM-ARMS-AT-SIDES"))
        self.assertFalse(inputs._as_expected({"outcome": "stopped", "codes": ["VM-NO-TEXTURE"]}, "VM-ARMS-DOWN VM-ARMS-AT-SIDES"))
        # a copy without its picture that built all the same was not stopped: the fault this looks for
        self.assertFalse(inputs._as_expected(same, "VM-NO-TEXTURE"))

    def test_two_builds_are_compared_by_what_does_not_number_vertices(self):
        from app import inputs
        movement = lambda name, labels: (self.root / f"{name}.json").write_text(json.dumps({"pieces": [{"label": n, "findings": []} for n in labels]})) and self.root / f"{name}.json"  # noqa: E731
        model = {"check": self.check("check_a", 1.2), "movement": movement("moves_a", [10, 20])}
        # the same pieces under other vertex numbers (a file saved again): nothing worse
        self.assertEqual(inputs._held(model, {"check": self.check("check_b", 1.2), "movement": movement("moves_b", [777, 888])}),
                         {"worse": [], "better": []})
        worse = inputs._held(model, {"check": self.check("check_c", 9.5), "movement": movement("moves_c", [1, 2, 3])})
        self.assertEqual(len(worse["worse"]), 2)
        self.assertTrue(worse["worse"][0].startswith("rig check"))
        self.assertIn("3 pieces flagged, the model's own build has 2", worse["worse"][1])

    def test_a_copy_is_a_new_character_with_the_models_options_and_nothing_else_of_it(self):
        from app import inputs
        base = self.made.project({"options": {}, "triangles": 40000}, chest=True, eyes="none")
        base.data["pieces"] = [{"name": "cape", "point": [0, 0, 1], "count": 9, "role": "swing_cloth"}]
        model = self.root / "copy one.glb"
        model.write_bytes(b"glTF")
        copy = inputs._character(self.root / "out" / "inputs" / "turn_x_90", model, base, "turn_x_90")
        self.assertEqual((copy.name, copy.out), ("Test_turn_x_90", self.root / "out" / "inputs" / "turn_x_90"))
        self.assertEqual((copy.data["options"]["chest"], copy.data["options"]["eyes"]), (True, "none"))
        self.assertEqual((copy.data["pieces"], copy.data.get("face"), copy.data["joints"]), ([], None, {}))
        self.assertEqual(manifest.path_of(copy.data, "source"), model)
        # a rigged FBX is its own rig there too
        rigged = self.root / "copy two.fbx"
        rigged.write_bytes(b"Kaydara FBX ... J_Bip_C_Hips ...")
        self.assertEqual(manifest.path_of(inputs._character(self.root / "out" / "inputs" / "as_fbx", rigged, base, "as_fbx").data, "rig"), rigged)

    def test_the_copies_it_asks_for_are_the_ones_the_script_makes(self):
        # the names the command line takes (--only) and the codes a copy is held to are make_variants.py's own
        import re
        script = (REPO / "scripts" / "make_variants.py").read_text(encoding="utf-8")
        names = re.findall(r'^        \("([a-z0-9_]+)", ', script, re.M)
        self.assertGreaterEqual(len(names), 17)
        for code in set(re.findall(r"VM-[A-Z-]+", script)):
            self.assertIsNotNone(errors.describe(code), code)


if __name__ == "__main__":
    unittest.main()
