#!/usr/bin/env python3
r"""Send Ctrl-C to the console of another process (Windows): python tools/win_ctrlc.py PID

The target must own its console (start it with subprocess.CREATE_NEW_CONSOLE), otherwise the event also reaches the caller's process group. This helper detaches from its own console, attaches to the
target's, ignores the event itself and generates CTRL_C_EVENT for everything attached to that console. llama-server turns it into a graceful shutdown (and compute-sanitizer wrapped around it then prints its
ERROR SUMMARY when the target exits). Exit code 0 when the event was generated, 2 otherwise."""
import ctypes
import sys
import time


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print(__doc__)
        return 2
    pid = int(sys.argv[1])
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.FreeConsole()
    if not k.AttachConsole(pid):
        print("AttachConsole(%d) failed: %d" % (pid, ctypes.get_last_error()))
        return 2
    k.SetConsoleCtrlHandler(None, True)
    ok = k.GenerateConsoleCtrlEvent(0, 0)
    time.sleep(1.0)
    k.FreeConsole()
    if not ok:
        print("GenerateConsoleCtrlEvent failed: %d" % ctypes.get_last_error())
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
