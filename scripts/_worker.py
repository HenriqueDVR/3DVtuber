"""One Blender that runs several of a build's steps, each as a Blender of its own would have run it.

Started by the app's engine (app/engine.py, `_Worker`) after scripts/_startup.py, with no file open:

    blender -b --factory-startup -P scripts/_startup.py -P scripts/_worker.py

A Blender takes about 1.6 s to start and switch the VRM Add-on on before it does anything, and a
build started one for every step: thirteen to seventeen of them, most for a step of two or three
seconds. This one stays, and is told each step on its standard input, a line of JSON:

    {"blend": the .blend the step starts from, "argv": the command line a Blender of the step's own
     would have had, "run": [["expr", code] or ["file", script], ...] in the order that command line
     runs them, "log": the step's log, "done": the file to write when it is over}

and answers by writing `done`: {"code": the exit code that Blender would have ended with}.

What a step can leave behind in a Blender is put back before the next one: the file (each opens its
own), the environment (the engine's switches travel in it), sys.argv and sys.path, save and load
handlers a script hung up, and the scripts' own modules, so each is read afresh. Each script runs in
a `__main__` of its own, as -P and --python-expr give it. Everything the step writes, Python's and
Blender's own, goes to its log. An error or a sys.exit ends the step with the code Blender would
have exited with; the worker itself goes on, and stops when its input closes (the engine has gone).
"""
import ctypes
import gc
import json
import os
import sys
import traceback
import types

import bpy

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def _flush():
    """What Python and Blender's own code have written and not yet handed over."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    try:
        (ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)).fflush(None)
    except (OSError, AttributeError):
        pass


def _into(log):
    """Send all this process writes to `log` (after what is there); returns what undoes it."""
    _flush()
    kept = os.dup(1), os.dup(2)
    target = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
    os.dup2(target, 1)
    os.dup2(target, 2)
    os.close(target)

    def back():
        _flush()
        os.dup2(kept[0], 1)
        os.dup2(kept[1], 2)
        os.close(kept[0])
        os.close(kept[1])
    return back


def _as_main(code, name):
    """Run compiled code in a `__main__` of its own, as Blender runs each -P and --python-expr."""
    main, before = types.ModuleType("__main__"), sys.modules.get("__main__")
    main.__dict__.update({"__file__": name, "__builtins__": __builtins__})
    sys.modules["__main__"] = main
    try:
        exec(code, main.__dict__)
    finally:
        sys.modules["__main__"] = before


def _run(order):
    """The step; returns the exit code its own Blender would have ended with."""
    try:
        if order.get("blend"):
            if bpy.ops.wm.open_mainfile(filepath=order["blend"]) != {"FINISHED"}:
                print(f"Error: Cannot read file {order['blend']!r}")
                return 1
        sys.argv[:] = order["argv"]
        for kind, what in order["run"]:
            if kind == "file":
                with open(what, "rb") as handle:
                    _as_main(compile(handle.read(), what, "exec"), what)
            else:
                _as_main(compile(what, "<blender string>", "exec"), "<blender string>")
        return 0
    except SystemExit as leaving:
        # as Python ends on one: nothing or 0 is a good end, a number is the code, words are said and are 1
        if leaving.code is None or isinstance(leaving.code, int):
            return int(leaving.code or 0)
        print(leaving.code, file=sys.stderr)
        return 1
    except BaseException:  # noqa: BLE001 -- whatever a step raises ends that step, not the worker
        traceback.print_exc()
        return 1


def _step(order):
    environment, argv, path = dict(os.environ), list(sys.argv), list(sys.path)
    modules = set(sys.modules)
    handlers = {name: list(getattr(bpy.app.handlers, name)) for name in dir(bpy.app.handlers)
                if isinstance(getattr(bpy.app.handlers, name), list)}
    back = _into(order["log"])
    try:
        code = _run(order)
    finally:
        try:
            for name, hung in handlers.items():
                getattr(bpy.app.handlers, name)[:] = hung
            for key in set(os.environ) - set(environment):
                del os.environ[key]
            os.environ.update(environment)
            sys.argv[:], sys.path[:] = argv, path
            # a script another script imported (fix_mouth.py reads add_face_tracking.py) is read anew next time
            for name in set(sys.modules) - modules:
                if os.path.dirname(os.path.abspath(getattr(sys.modules[name], "__file__", None) or "")) == SCRIPTS:
                    del sys.modules[name]
            gc.collect()
        finally:
            back()
    return code


def main():
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return  # the engine has closed our input, or is gone: so are we
        order = json.loads(line)
        if order.get("quit"):
            return
        code = _step(order)
        with open(order["done"] + ".part", "w", encoding="utf-8") as handle:
            json.dump({"code": code}, handle)
        os.replace(order["done"] + ".part", order["done"])


main()
