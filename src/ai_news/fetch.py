import time

import requests

from .observability import get_logger

LOGGER = get_logger("fetch")
USER_AGENT = "Mozilla/5.0 (compatible; AI-News-Reporter/2.0; +https://github.com/)"
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """Raised when a URL could not be retrieved after every attempt."""


def get(url: str, timeout: float = 15.0, attempts: int = 3, backoff: float = 1.5) -> requests.Response:
    """GET with bounded retries. Every attempt is logged so failures stay visible."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if response.status_code in RETRY_STATUS and attempt < attempts:
                LOGGER.warning("retryable status %s from %s (attempt %s/%s)", response.status_code, url, attempt, attempts)
                last_error = requests.HTTPError(f"status {response.status_code}")
                time.sleep(backoff ** attempt)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < attempts:
                LOGGER.warning("fetch failed for %s (attempt %s/%s): %s", url, attempt, attempts, error)
                time.sleep(backoff ** attempt)
    LOGGER.error("fetch gave up on %s after %s attempts: %s", url, attempts, last_error)
    raise FetchError(str(last_error)) from last_error
