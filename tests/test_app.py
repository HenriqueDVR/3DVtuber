"""The app's engine plans a character's steps from its manifest, and its server hands out only what it should."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import cli, engine, server  # noqa: E402
from app.engine import Project, manifest  # noqa: E402


class Engine(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "character.json"
        data = manifest.new("model.glb", "Test")
        data["out_dir"] = "out"
        data["face"] = "face.json"
        manifest.save(data, self.path)

    def tearDown(self):
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
        self.assertNotIn("mouth", runs)

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


class Server(unittest.TestCase):
    def test_serves_only_model_and_output_files_inside_the_repository(self):
        inside = REPO / "viewer" / "index.html"
        self.assertEqual(server._served(inside), "viewer/index.html")
        for path in (REPO / "out" / ".." / "CLAUDE.md", REPO / "scripts" / "make_vrm.py", REPO / ".git" / "config",
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

    def test_refuses_a_name_with_a_path(self):
        with self.assertRaisesRegex(ValueError, "path"):
            server.api("init", {}, {"model": "m.glb", "name": "../evil"})


if __name__ == "__main__":
    unittest.main()
