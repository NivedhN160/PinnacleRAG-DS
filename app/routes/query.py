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
        domain = request.domain.lower().strip()
        if request.mode == "agent":
            result = agent_loop.run(request.question, domain=domain)
        else:
            result = query_pipeline.run(
                request.question, 
                domain=domain,
                rewrite=request.rewrite,
                check_faithfulness=request.check_faithfulness
            )
            
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        error_str = str(e).lower()
        if "budget" in error_str or "limit" in error_str:
            raise HTTPException(status_code=429, detail=f"Rate limit or budget exceeded: {str(e)}")
        if "invalid" in error_str:
            raise HTTPException(status_code=400, detail=f"Bad request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
