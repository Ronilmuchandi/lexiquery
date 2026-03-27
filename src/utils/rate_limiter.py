# =============================================================
# FILE: src/utils/rate_limiter.py
# PURPOSE: Rate limiting to avoid hitting API limits
# =============================================================

import time


class RateLimiter:
    """
    Simple rate limiter to control API call frequency.
    
    WHAT THIS DOES:
    Ensures we don't make too many API calls too quickly,
    which could cause rate limit errors from Groq.
    """

    def __init__(self, calls_per_minute: int = 30):
        """
        Initialize rate limiter.
        
        INPUT:
            calls_per_minute → maximum API calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limits.
        Call this before every API request.
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()


# Global rate limiter instance
# Import this wherever you make API calls
groq_rate_limiter = RateLimiter(calls_per_minute=30)