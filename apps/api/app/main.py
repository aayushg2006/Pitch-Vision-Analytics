from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .data_access import AnalyticsRepository, get_repository
from .models import (
    CaptainAdviceRequest,
    CaptainResponse,
    DashboardSummaryResponse,
    DataProofResponse,
    HeadToHeadRequest,
    HeadToHeadResponse,
    ImpactOptimizationRequest,
    ImpactOptimizationResponse,
    OptionsResponse,
    PartnershipRequest,
    PartnershipResponse,
    PressureFilter,
    PressureOverviewResponse,
    WinPredictRequest,
    WinPredictResponse,
)
from .services.captain import chat_with_captain, get_captain_advice, optimize_impact
from .services.dashboard import get_dashboard_summary, get_data_proof, get_options
from .services.player_analytics import get_head_to_head, get_partnership, get_pressure_stats
from .services.predictor import predict


app = FastAPI(title="Pitch Vision Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(repository: AnalyticsRepository = Depends(get_repository)) -> DashboardSummaryResponse:
    return get_dashboard_summary(repository)


@app.get("/api/v1/options", response_model=OptionsResponse)
def options(repository: AnalyticsRepository = Depends(get_repository)) -> OptionsResponse:
    return get_options(repository)


@app.get("/api/v1/data-proof", response_model=DataProofResponse)
def data_proof(repository: AnalyticsRepository = Depends(get_repository)) -> DataProofResponse:
    return get_data_proof(repository)


@app.post("/api/v1/predict", response_model=WinPredictResponse)
def win_predict(payload: WinPredictRequest, repository: AnalyticsRepository = Depends(get_repository)) -> WinPredictResponse:
    return predict(repository, payload)


@app.post("/api/v1/matchups/head-to-head", response_model=HeadToHeadResponse)
def head_to_head(
    payload: HeadToHeadRequest, repository: AnalyticsRepository = Depends(get_repository)
) -> HeadToHeadResponse:
    return get_head_to_head(repository, payload)


@app.post("/api/v1/matchups/partnership", response_model=PartnershipResponse)
def partnership(
    payload: PartnershipRequest, repository: AnalyticsRepository = Depends(get_repository)
) -> PartnershipResponse:
    return get_partnership(repository, payload)


@app.post("/api/v1/clutch", response_model=PressureOverviewResponse)
def clutch(payload: PressureFilter, repository: AnalyticsRepository = Depends(get_repository)) -> PressureOverviewResponse:
    return get_pressure_stats(repository, payload)


@app.post("/api/v1/captain/advice", response_model=CaptainResponse)
def captain_advice(
    payload: CaptainAdviceRequest, repository: AnalyticsRepository = Depends(get_repository)
) -> CaptainResponse:
    return get_captain_advice(repository, payload)


@app.post("/api/v1/captain/chat", response_model=CaptainResponse)
def captain_chat(
    payload: CaptainAdviceRequest, repository: AnalyticsRepository = Depends(get_repository)
) -> CaptainResponse:
    return chat_with_captain(repository, payload)


@app.post("/api/v1/impact/optimize", response_model=ImpactOptimizationResponse)
def impact_optimize(
    payload: ImpactOptimizationRequest, repository: AnalyticsRepository = Depends(get_repository)
) -> ImpactOptimizationResponse:
    return optimize_impact(repository, payload)
