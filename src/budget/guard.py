"""
Budget guard for PinnacleRAG-DS.
Tracks API usage and throws Hard Stop if budget exceeded.
"""
from fastapi import HTTPException
from config.settings import Settings

class BudgetGuard:
    def __init__(self, settings: Settings):
        self.max_calls = settings.max_llm_calls
        self.current_calls = 0

    def check_budget(self) -> None:
        if self.current_calls >= self.max_calls:
            raise HTTPException(
                status_code=429, 
                detail="budget_exceeded"
            )

    def record_call(self) -> None:
        self.current_calls += 1

    def get_remaining(self) -> int:
        return max(0, self.max_calls - self.current_calls)
