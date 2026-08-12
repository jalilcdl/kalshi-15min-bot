@echo off
REM ===========================================================================
REM  Supervisor entry point for live_trader.py (Phase 3, DEMO).
REM
REM  WHY THIS EXISTS
REM  On 2026-08-11 the trader ran as a child of an agent session's shell. When
REM  that session ended the process tree went with it: dead at 22:38 UTC, no
REM  crash, no traceback, down for 21 hours while an open 25-contract position
REM  settled unattended. Anything meant to run unattended must not be parented
REM  to whatever happened to start it.
REM
REM  WHY pythonw.exe AND NOT python.exe
REM  Task Scheduler runs this in the INTERACTIVE console session (Session 1) --
REM  the same console any other tooling on this machine uses. A console control
REM  event (Ctrl+C / Ctrl+Break) is delivered to EVERY process attached to that
REM  console, so an unrelated command being interrupted kills the trader too.
REM  Observed 2026-08-12: live_run.err ends in "^C^C" and the trader vanished
REM  mid-position with no exception of its own.
REM
REM  pythonw.exe is the windowless interpreter -- it has no console attached, so
REM  console control events cannot reach it. Redirection below still captures
REM  stdout/stderr because cmd sets up the handles before pythonw inherits them.
REM
REM  MORE ROBUST STILL (needs a decision, not code): configuring the task to
REM  "run whether the user is logged on or not" puts it in Session 0, isolated
REM  from the interactive desktop entirely, and also survives logout. That
REM  requires storing account credentials, so it is deliberately left for the
REM  account owner to set rather than automated here.
REM
REM  Task Scheduler fires this every 5 minutes. live_trader takes an OS-level
REM  singleton lock and the task uses IgnoreNew, so a tick while healthy is a
REM  no-op and a tick after a death is a restart: self-healing, not one-shot.
REM
REM  Full interpreter path on purpose: a scheduled task does not inherit the
REM  interactive PATH, and "python" may simply not resolve.
REM ===========================================================================
cd /d "C:\Users\jalil\OneDrive\Desktop\kalshi-15min-bot"
"C:\Users\jalil\AppData\Local\Programs\Python\Python312\pythonw.exe" live_trader.py >> live_run.log 2>> live_run.err
