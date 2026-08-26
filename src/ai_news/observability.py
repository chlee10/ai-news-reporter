import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER = "ai_news"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _force_utf8_console() -> None:
    """Korean output must not crash the run on a cp949 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def configure_logging(verbose: bool = False, log_path: Path | None = Path("logs/ai-news.log")) -> logging.Logger:
    """Attach stderr and rotating-file handlers once. Silence is never an acceptable failure mode."""
    _force_utf8_console()
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            rotating = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            rotating.setFormatter(formatter)
            logger.addHandler(rotating)
        except OSError:
            logger.warning("log file unavailable at %s; continuing with stderr only", log_path)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
