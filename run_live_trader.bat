@echo off
REM ===========================================================================
REM  Supervisor entry point for live_trader.py (Phase 3, DEMO).
REM
REM  WHY THIS EXISTS
REM  On 2026-08-11 the trader was launched with nohup as a child of an agent
REM  session's shell. When that session ended, the process tree went with it and
REM  the trader died silently at 22:38 UTC -- no crash, no traceback, just gone,
REM  and it stayed down for 21 hours. Anything meant to run unattended must not
REM  be parented to whatever happened to start it.
REM
REM  Windows Task Scheduler runs this every few minutes. live_trader takes an
REM  OS-level singleton lock, so:
REM    - if a healthy instance is running, this exits immediately and quietly
REM      (it writes one line to live_supervisor.log and returns 0)
REM    - if the trader has died, the next tick restarts it
REM  That makes the supervision self-healing rather than one-shot.
REM
REM  Full python path on purpose: a scheduled task does not inherit the
REM  interactive PATH, and "python" may simply not resolve.
REM ===========================================================================
cd /d "C:\Users\jalil\OneDrive\Desktop\kalshi-15min-bot"
"C:\Users\jalil\AppData\Local\Programs\Python\Python312\python.exe" live_trader.py >> live_run.log 2>> live_run.err
