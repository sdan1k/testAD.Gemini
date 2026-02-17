// API функции для взаимодействия с бэкендом

export interface CaseResult {
  index: number;
  score: number;
  field_scores?: Record<string, number>;
  docId?: string;
  Violation_Type?: string;
  document_date?: string;
  FASbd_link?: string;
  FAS_division?: string;
  violation_found?: string;
  defendant_name?: string;
  defendant_industry?: string;
  ad_description?: string;
  ad_content_cited?: string;
  ad_platform?: string;
  violation_summary?: string;
  FAS_arguments?: string;
  legal_provisions?: string;
  thematic_tags?: string;
}

export interface SearchResponse {
  query: string;
  total_cases: number;
  results: CaseResult[];
  filters_applied?: Record<string, unknown>;
  message?: string;
}

export interface SubIndustry {
  name: string;
  count: number;
  sub_industries?: SubIndustry[];
}

export interface IndustryGroup {
  name: string;
  count: number;
  sub_industries: SubIndustry[];
}

export interface FilterOptions {
  years: number[];
  regions: string[];
  industries: string[];
  industry_groups: IndustryGroup[];
  region_groups?: { name: string; count: number; regions: string[] }[];
  article_groups?: { name: string; count: number; parts: { name: string; count: number }[] }[];
  region_groups: { name: string; count: number; regions: string[] }[];
  articles: string[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function searchCases(
  query: string,
  topK: number = 20,
  filters?: {
    year?: number[];
    region?: string[];
    industry?: string[];
    article?: string[];
  }
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/api/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      top_k: topK,
      ...filters,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Ошибка поиска');
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await fetch(`${API_BASE}/api/filters`);
  if (!response.ok) {
    throw new Error('Ошибка загрузки фильтров');
  }
  return response.json();
}
