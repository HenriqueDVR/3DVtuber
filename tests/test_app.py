"""The app's engine plans a character's steps from its manifest, and its server hands out only what it should."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import cli, engine, errors, server, vendor  # noqa: E402
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
