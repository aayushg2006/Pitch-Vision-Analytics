import type {
  CaptainAdviceRequest,
  CaptainResponse,
  DashboardSummaryResponse,
  DataProofResponse,
  HeadToHeadRequest,
  HeadToHeadResponse,
  ImpactOptimizationRequest,
  ImpactOptimizationResponse,
  InsightsResponse,
  OptionsResponse,
  PartnershipRequest,
  PartnershipResponse,
  PressureFilter,
  PressureOverviewResponse,
  WinPredictRequest,
  WinPredictResponse,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export const chartUrl = (name: string) => `${API_BASE_URL}/insights/charts/${name}`;

export const api = {
  getDashboardSummary: () => request<DashboardSummaryResponse>("/dashboard/summary"),
  getOptions: () => request<OptionsResponse>("/options"),
  getDataProof: () => request<DataProofResponse>("/data-proof"),
  getInsights: () => request<InsightsResponse>("/insights"),
  predict: (payload: WinPredictRequest) =>
    request<WinPredictResponse>("/predict", { method: "POST", body: JSON.stringify(payload) }),
  headToHead: (payload: HeadToHeadRequest) =>
    request<HeadToHeadResponse>("/matchups/head-to-head", { method: "POST", body: JSON.stringify(payload) }),
  partnership: (payload: PartnershipRequest) =>
    request<PartnershipResponse>("/matchups/partnership", { method: "POST", body: JSON.stringify(payload) }),
  clutch: (payload: PressureFilter) =>
    request<PressureOverviewResponse>("/clutch", { method: "POST", body: JSON.stringify(payload) }),
  captainAdvice: (payload: CaptainAdviceRequest) =>
    request<CaptainResponse>("/captain/advice", { method: "POST", body: JSON.stringify(payload) }),
  impactOptimize: (payload: ImpactOptimizationRequest) =>
    request<ImpactOptimizationResponse>("/impact/optimize", { method: "POST", body: JSON.stringify(payload) }),
};

