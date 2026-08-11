"""
Single-instance lock, so two copies of a trader can never run at once.

WHY THIS EXISTS (2026-08-11): two live_trader.py processes ran concurrently for
~13 minutes, both evaluating and both placing orders. A `kill` issued from Git
Bash silently failed against the Windows process, so what was meant to be a
restart produced a duplicate.

Nothing already in the system caught it:
  - LIVE_MAX_CONTRACTS is enforced PER ORDER, not per account. Two processes at
    the cap means 2x the intended exposure.
  - client_order_id idempotency protects against retry-after-timeout, where the
    SAME id is resent. Two independent processes mint different uuids, so the
    exchange sees two legitimate distinct orders.
Only luck prevented a double position: every order that session happened to
miss its fill.

Implementation is an OS-level advisory lock on a file, NOT a bare pid file. A
pid file goes stale on a crash and then either blocks a legitimate start or
gets ignored; an OS lock is released by the kernel when the holding process
dies, however it dies. The pid is written into the file too, but purely as a
human-readable hint for whoever is looking at it.
"""
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class AlreadyRunning(RuntimeError):
    pass


class SingleInstance:
    """Context manager. Raises AlreadyRunning if another holder is alive.

        with SingleInstance("live_trader"):
            main_loop()
    """

    def __init__(self, name: str, directory: Path | None = None):
        d = directory or Path(__file__).parent
        self.path = d / f".{name}.lock"
        self._fh = None

    def acquire(self):
        # Open (not truncate) so a failed lock attempt cannot destroy the pid
        # written by the process that legitimately holds it.
        self._fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Report the holder's pid if we can, but NEVER let that reporting
            # fail the lock check: on Windows the locked byte range is also
            # unreadable from another process, so a naive read() here raises
            # PermissionError and masks the real (correct) "already running"
            # result. The lock decision must not depend on the diagnostic.
            holder = "unknown"
            try:
                self._fh.seek(0)
                holder = self._fh.read().strip() or "unknown"
            except OSError:
                pass
            finally:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
            raise AlreadyRunning(
                f"another instance already holds {self.path} (pid {holder}). "
                "Refusing to start a second copy -- two traders would double the "
                "account's exposure while each individually respects the size cap."
            )
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def release(self):
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()
        return False
