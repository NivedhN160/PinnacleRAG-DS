from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse
from config.settings import get_settings
from src.budget.guard import BudgetGuard
from src.pipeline.query_pipeline import QueryPipeline
from src.pipeline.agent_loop import AgentLoop

router = APIRouter()

# Simple dependency injection
settings = get_settings()
budget_guard = BudgetGuard(settings)
query_pipeline = QueryPipeline(settings, budget_guard)
agent_loop = AgentLoop(settings, budget_guard, query_pipeline)

@router.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        if request.mode == "agent":
            result = agent_loop.run(request.question, domain=request.domain)
        else:
            result = query_pipeline.run(request.question, domain=request.domain)
            
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
