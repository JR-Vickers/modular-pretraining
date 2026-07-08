"""
Preemption handler for Slurm jobs.

When Slurm preempts a low-priority job, it sends SIGTERM before killing
the process (configurable grace period via --signal=B:SIGTERM@180).

This module catches SIGTERM, sets a flag that training loops can check,
and then restores the default handler so that a subsequent SIGTERM (or
the same one, if the process is stuck in a blocking call) actually kills
the process.  This prevents zombie workers from holding port 29500 or
/dev/shm/nccl-* after torchrun tears down a failed job.

Usage:
    from src.run.util.preemption import setup_preemption, is_preempted

    setup_preemption()  # Call once at start of training

    for step in training_loop:
        ...
        if should_save(step, total_steps, checkpoint_freq):
            save_checkpoint(...)
            if is_preempted():
                sys.exit(0)  # IMPORTANT: exit after saving, _preempted stays True
"""

import os
import signal
import logging
import threading

_preempted = False
_force_exit_timer: threading.Timer | None = None
_logger = logging.getLogger(__name__)

# Slurm typically sends SIGTERM ~180s before SIGKILL.  Leave headroom for the
# process to finish writing a checkpoint (large models can take >30s).
GRACE_SECONDS = float(os.environ.get("PREEMPT_GRACE_SECONDS", 170.0))


def is_preempted() -> bool:
    """Check if preemption has been requested via SIGTERM."""
    return _preempted


def cancel_forced_exit() -> None:
    """Cancel the pending forced-exit timer (call after checkpoint is safely on disk)."""
    global _force_exit_timer
    if _force_exit_timer is not None:
        _force_exit_timer.cancel()
        _force_exit_timer = None


def _handler(signum: int, frame: object) -> None:
    """SIGTERM handler: set flag, restore default, schedule forced exit."""
    global _preempted, _force_exit_timer
    _preempted = True

    # Restore defaults so a second signal kills immediately
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    _logger.warning(
        f"Received signal {signum} (SIGTERM). "
        "Preemption requested — will save checkpoint and exit at next opportunity. "
        f"Process will force-exit in {GRACE_SECONDS:.0f}s if still alive."
    )

    # If the process is stuck in a blocking call (barrier, all_reduce, etc.)
    # it will never reach the is_preempted() check.  Schedule a hard exit
    # so we don't leave zombie workers holding resources.
    def _force_exit():
        _logger.error("Preemption grace period expired — forcing exit")
        os._exit(1)

    _force_exit_timer = threading.Timer(GRACE_SECONDS, _force_exit)
    _force_exit_timer.daemon = True
    _force_exit_timer.start()


def setup_preemption() -> None:
    """Register SIGTERM/SIGINT handler for graceful checkpoint-and-exit."""
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    _logger.info("Preemption handler registered (SIGTERM, SIGINT)")
