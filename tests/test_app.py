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

from app import cli, engine, errors, paint, server, update, vendor  # noqa: E402
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
        with mock.patch.object(report, "REPO", self.root), mock.patch.object(engine, "setup", lambda: {"version": "5.2.2", "checks": [
                {"name": "VRM Add-on", "state": "bad", "what": "the VRM Add-on isn't installed", "fix": "..."}]}):
            (self.root / "out").mkdir(exist_ok=True)
            (self.root / "out" / "app_errors.log").write_text(f"Traceback in {self.root}")
            made = report.make(None, note="won't start", about="install failed")
        self.assertEqual(made["headline"][0].split(" | ")[0], "**No character open**")
        self.assertIn("Setup: the VRM Add-on isn't installed", made["headline"])
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

    def test_a_refused_automatic_iris_slides_instead(self):
        project = self.project({"options": {}, "triangles": 40000, "parts": {"eye_inserts": True}})
        self.assertEqual(project.options()["eyes"], "iris")
        refused = engine.StepFailed("eyes: iris over the lids", errors.describe("VM-EYES-REFUSED"))
        # first the model's own iris discs, then the painted patch, then nothing more
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "modelled")
        args, _ = project._arguments("eyes", self.root / "11_eyes", None)
        self.assertIn("--modelled", [str(a) for a in args])
        self.assertTrue(project._eyes_fall_back("eyes", refused))
        self.assertEqual(project.options()["eyes"], "slide")
        args, _ = project._arguments("eyes", self.root / "11_eyes", None)
        self.assertIn("--slide", [str(a) for a in args])
        self.assertFalse(project._eyes_fall_back("eyes", refused))
        # kept for the next build of the same model
        self.assertEqual(Project(self.path, blender="blender").options()["eyes"], "slide")
        # an iris the user chose stays an iris, and other failures aren't retried
        chosen = self.project({"options": {}, "triangles": 40000}, eyes="iris")
        self.assertFalse(chosen._eyes_fall_back("eyes", refused))
        self.assertFalse(project._eyes_fall_back("mouth", refused))

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

    def test_the_finished_vrm_is_sent_to_warudo_once_and_never_written_over(self):
        project = self.project()
        library = self.root / "SteamLibrary"
        real, engine.steam_libraries = engine.steam_libraries, lambda: [self.root / "Steam", library]
        try:
            vrm = project.out / "runs" / "20260104-100000" / "13_export.vrm"
            vrm.parent.mkdir(parents=True)
            vrm.write_bytes(b"x" * 1000)
            with self.assertRaisesRegex(engine.StepFailed, "build first"):
                project.send_to_warudo()
            project.state["steps"]["export"] = {"status": "done", "vrm": str(vrm)}
            with self.assertRaisesRegex(FileNotFoundError, "Warudo wasn't found"):
                project.send_to_warudo()
            characters = library / engine.WARUDO_CHARACTERS
            characters.mkdir(parents=True)
            sent = project.send_to_warudo()
            self.assertEqual((sent["name"], sent["already"]), ("Test_20260104-100000.vrm", False))
            self.assertEqual([p.name for p in characters.iterdir()], ["Test_20260104-100000.vrm"])
            self.assertTrue(project.send_to_warudo()["already"])
            (characters / sent["name"]).write_bytes(b"someone else's")
            with self.assertRaisesRegex(engine.StepFailed, "a different file"):
                project.send_to_warudo()
            self.assertEqual((characters / sent["name"]).read_bytes(), b"someone else's")
            # a folder given by hand wins over Steam's, and only a folder is taken
            mine = self.root / "Portable Warudo" / "Characters"
            with self.assertRaisesRegex(ValueError, "isn't a folder"):
                engine.use_warudo(str(mine))
            mine.mkdir(parents=True)
            self.assertEqual(engine.use_warudo(str(mine)), {"warudo": str(mine)})
            self.assertEqual(Path(project.send_to_warudo()["sent"]).parent, mine)
            self.assertEqual(engine.use_warudo(""), {"warudo": str(characters)})
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
        for route in ("run", "prune", "approve", "cancel", "vendor", "sheet", "face_guess", "send_warudo", "set", "bundle", "report", "report_send", "report_address"):
            probe = Probe(f"/api/{route}?manifest=x")
            probe.do_GET()
            self.assertEqual(probe.said, [405], route)
        for route in server.READ_ONLY:
            probe = Probe(f"/api/{route}")
            probe.do_GET()
            self.assertEqual(probe.said, [route])

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
        # in both looks (Setup): the arcade one adds its own screens, the classic one must stay as it was
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
        self.assertIn('class="page on" id="setup"', new.stdout)
        self.assertNotIn('id="start-go" hidden', new.stdout)
        self.assertEqual([line for line in first.stderr.splitlines() if "CONSOLE" in line and "Uncaught" in line], [])
        for mark in ('class="page on" id="setup"', "Blender 4.2 or later wasn't found", "Get Blender", "App version"):
            self.assertTrue(mark in first.stdout, f"first start: {mark}")
        self.assertNotIn('class="page on" id="home"', first.stdout)
        for look, done in pages.items():
            page = done.stdout
            errors_said = [line for line in done.stderr.splitlines() if "CONSOLE" in line and ("Uncaught" in line or "SyntaxError" in line)]
            self.assertEqual(errors_said, [], look)
            # each is drawn by its module's last lines, or by main.js after every import ran
            for mark in ('data-tool="brush"', "New character", 'class="page on" id="home"', "Open or create a character first"):
                self.assertTrue(mark in page, f"not drawn in the {look} look: {mark}")
            # the arcade look's stage, roster and menus are there in it alone
            for mark in ('id="a-bg"', 'class="slot new', 'id="menu"', 'id="a-crumbs"'):
                self.assertEqual(mark in page, look == "arcade", f"{look} look: {mark}")


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


if __name__ == "__main__":
    unittest.main()
