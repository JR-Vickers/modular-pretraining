"""
Logging utilities for multiprocessing-safe logging with tqdm support.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from tqdm.auto import tqdm
from src.run.util.distributed import is_main_process


class DistributedFilter(logging.Filter):
    """Filter that only allows logging from main process in distributed training."""
    
    def filter(self, record):
        # Allow logs explicitly marked to emit from all ranks
        if getattr(record, "_all_ranks", False):
            return True
        return is_main_process()


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that plays nicely with tqdm progress bars."""
    
    def emit(self, record):
        try:
            # Use tqdm.write to avoid breaking progress bars
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def _create_formatter(process_id: Optional[int] = None) -> logging.Formatter:
    """Create a logging formatter with optional process ID prefix."""
    if process_id is not None:
        format_str = f'[GPU {process_id}] [%(asctime)s] [%(levelname)s] %(message)s'
    else:
        format_str = '[%(asctime)s] [%(levelname)s] %(message)s'
    return logging.Formatter(format_str, datefmt='%H:%M:%S')


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    distributed_aware: bool = True,
    process_id: Optional[int] = None,
    multiprocessing_queue: Optional[mp.Queue] = None,
) -> logging.Logger:
    """
    Set up a logger instance with optional file output and multiprocessing support.
    
    In distributed training, automatically filters logs to only show from rank 0
    to avoid duplicate logging across GPUs.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        distributed_aware: If True, automatically filter logs from non-main processes
        process_id: Optional process/GPU ID for multiprocessing context
        multiprocessing_queue: Optional queue for multiprocessing logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Add distributed filter if enabled (but not when using multiprocessing queue)
    if distributed_aware and multiprocessing_queue is None:
        logger.addFilter(DistributedFilter())
    
    # Create formatter
    formatter = _create_formatter(process_id)
    
    if multiprocessing_queue is not None:
        # Use queue handler for multiprocessing
        queue_handler = QueueHandler(multiprocessing_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)
    else:
        # Console handler using TqdmLoggingHandler
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified - only create on main process to avoid
        # empty log files from non-main ranks in DDP mode
        if log_file is not None and (not distributed_aware or is_main_process()):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_multiprocess_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> tuple[mp.Queue, QueueListener, Any]:
    """
    Set up multiprocessing-safe logging with a queue.
    
    Args:
        log_file: Optional path to log file
        level: Logging level for handlers (default: INFO)
    
    Returns:
        Tuple of (log_queue, queue_listener, manager) - the listener must be started/stopped
        and manager must be kept alive
    """
    # Use Manager().Queue() for spawn compatibility
    manager = mp.Manager()
    log_queue = manager.Queue()
    
    # Create formatter
    formatter = _create_formatter()
    
    # Set up handlers for the queue listener
    handlers = []
    
    # Console handler
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Create queue listener
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    
    return log_queue, queue_listener, manager


def get_tqdm_kwargs(logger: logging.Logger, **kwargs: object) -> dict[str, object]:
    """
    Get tqdm kwargs based on logger level.

    Progress bars are only shown on the main process in DEBUG mode.
    In INFO mode or higher, progress bars are disabled entirely.
    """
    # Ensure tqdm writes to stderr to match logging handler
    kwargs.setdefault('file', sys.stderr)

    # Disable progress bars unless in DEBUG mode on main process
    kwargs['disable'] = not (logger.isEnabledFor(logging.DEBUG) and is_main_process())

    return kwargs
