"""The app's engine plans a character's steps from its manifest, and its server hands out only what it should."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import cli, engine, errors, server  # noqa: E402
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
                                             "finished": None, "approved": False, "picture": None})
        self.finished(project, eyes={"status": "failed", "finished": "2026-01-02T10:00:00"}, tpose={"status": "stale"})
        pictures = project.out / "renders" / "20260102-100000_09_tpose_raised"
        pictures.mkdir(parents=True)
        for view in ("back", "front"):
            (pictures / f"raised_{view}.png").write_bytes(b"")
        summary = project.summary()
        self.assertTrue(summary["built"])
        self.assertEqual((summary["failed"], summary["stale"]), (["eyes"], ["tpose"]))
        self.assertEqual(Path(summary["picture"]).name, "raised_front.png")

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


    def test_a_half_written_state_falls_back_to_its_backup(self):
        project = self.project()
        project._record("build", {"status": "done", "blend": "a.blend"})
        project._record("hair", {"status": "done", "blend": "b.blend"})
        project.state_path.write_text('{"steps": {"bui', encoding="utf-8")  # a crash mid-write
        self.assertEqual(Project(self.path).state["steps"]["build"]["blend"], "a.blend")
        project.state_path.with_name("state.json.bak").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(engine.StepFailed, "no usable backup"):
            Project(self.path)

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

    def test_shows_only_files_the_build_made(self):
        for path in (REPO / "CLAUDE.md", REPO / "out" / ".." / "scripts" / "make_vrm.py", REPO.parent):
            with self.assertRaisesRegex(ValueError, "only files the build made"):
                server._reveal(path)

    def test_render_folders_name_their_run_and_pose(self):
        match = lambda name: server.RENDER_FOLDER.match(name).groupdict()  # noqa: E731
        self.assertEqual(match("20260102-100000_09_tpose_t-pose"), {"run": "20260102-100000", "blend": "09_tpose", "pose": "t-pose"})
        self.assertEqual(match("09_tpose_raised"), {"run": None, "blend": "09_tpose", "pose": "raised"})

    def test_refuses_a_name_with_a_path(self):
        with self.assertRaisesRegex(ValueError, "path"):
            server.api("init", {}, {"model": "m.glb", "name": "../evil"})


if __name__ == "__main__":
    unittest.main()
