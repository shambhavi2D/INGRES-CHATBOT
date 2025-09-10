import os
import json
import re
import math
import difflib
import warnings
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Google Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

warnings.filterwarnings("ignore")

# --------------- ENV & GLOBALS ---------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA1_NI-CkG7CbKqBOdQ9Y5rJV8YtnMlE1Q").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro").strip() or "gemini-pro"

CANDIDATE_DATA_PATHS = list(
    filter(
        None,
        [
            os.getenv("DATA_PATH", "").strip() or None,
            "dummy_groundwater_data.csv",
            "/mnt/data/CentralReport1757434009468_CLEAN.csv",
        ],
    )
)

# --------------- FASTAPI SCHEMAS ---------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User query in natural language")
    top_k: int = Field(10, description="Top K rows for compare/listing")
    year: Optional[int] = Field(None, description="Force a specific year if present")
    debug: bool = Field(False, description="Return extra debug info")

class ChartSpec(BaseModel):
    type: str
    data: Dict[str, Any]
    options: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    intent: str
    metric: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    chart: Optional[ChartSpec] = None
    table: Optional[List[Dict[str, Any]]] = None
    narrative: str
    forecast: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None

# --------------- DATA LOADER & CLEANING ---------------

def _try_read(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
            return pd.read_excel(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return None

def load_dataset() -> pd.DataFrame:
    # Try to load dummy data first
    dummy_path = "dummy_groundwater_data.csv"
    if os.path.exists(dummy_path):
        try:
            df = pd.read_csv(dummy_path)
            print(f"[OK] Loaded data from: {dummy_path} | shape={df.shape}")
            return clean_dataframe(df)
        except Exception as e:
            print(f"[WARN] Failed to read {dummy_path}: {e}")
    
    # Fallback to other paths
    df: Optional[pd.DataFrame] = None
    for p in CANDIDATE_DATA_PATHS:
        df = _try_read(p)
        if df is not None:
            print(f"[OK] Loaded data from: {p} | shape={df.shape}")
            return clean_dataframe(df)
    
    # Create minimal dummy data if no dataset found
    print("[WARN] No dataset found, creating minimal dummy data")
    return create_minimal_dummy_data()

def create_minimal_dummy_data() -> pd.DataFrame:
    """Create minimal dummy data for testing"""
    data = {
        'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Gujarat', 'Rajasthan'],
        'District': ['Mumbai', 'Bangalore Urban', 'Chennai', 'Ahmedabad', 'Jaipur'],
        'Year': [2020, 2021, 2022, 2020, 2021],
        'Groundwater_Level_m': [5.2, 7.8, 6.1, 8.3, 12.5],
        'Annual_Recharge_ham': [1200, 950, 1100, 850, 600],
        'Extraction_ham': [1150, 920, 1050, 800, 580],
        'Stage_of_Ground_Water_Extraction_pct': [85, 92, 88, 78, 95],
        'Category': ['Semi-Critical', 'Critical', 'Semi-Critical', 'Safe', 'Critical']
    }
    return pd.DataFrame(data)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", " ", c2)
        new_cols.append(c2)
    df.columns = new_cols

    to_drop = [c for c in df.columns if c.lower().startswith("unnamed") and df[c].isna().all()]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": "", "null": ""})

    for c in df.columns:
        if c.lower() in {"id", "code", "s_no"}:
            continue
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() >= 0.3:  # Lower threshold for better conversion
                df[c] = s.fillna(0)
        except Exception:
            pass

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna("")

    return df

DF = load_dataset()

# --------------- SCHEMA INTROSPECTION HELPERS ---------------

def list_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def list_text_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == "object"]

def detect_year_column(df: pd.DataFrame) -> Optional[str]:
    year_indicators = ["year", "assessment_year", "yr", "annual", "fiscal_year"]
    for c in df.columns:
        if any(indicator in c.lower() for indicator in year_indicators):
            return c
    return None

def detect_geo_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    geo_mapping = {
        "state": ["state", "st", "province"],
        "district": ["district", "dist", "division"],
        "block": ["block", "taluk", "mandal", "tehsil"],
        "unit": ["unit", "assessment_unit", "region", "zone"]
    }
    
    found = {}
    for geo_type, keywords in geo_mapping.items():
        for c in df.columns:
            if any(keyword in c.lower() for keyword in keywords):
                found[geo_type] = c
                break
    return found

def detect_stage_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if ("stage" in cl and "extraction" in cl) or ("category" in cl and "water" in cl):
            return c
    return None

YEAR_COL = detect_year_column(DF) or "Year"
GEO = detect_geo_columns(DF)
STAGE_COL = detect_stage_column(DF) or "Category"
NUMERIC_COLS = list_numeric_columns(DF)
TEXT_COLS = list_text_columns(DF)

# Print dataset info for debugging
print(f"Dataset loaded: {DF.shape}")
print(f"Columns: {list(DF.columns)}")
print(f"Year column: {YEAR_COL}")
print(f"Geo columns: {GEO}")
print(f"Stage column: {STAGE_COL}")

# --------------- METRIC MATCHING (FUZZY) ---------------

ALIAS_HINTS = {
    "year": ["Year"],
    "state": ["State"],
    "district": ["District"],
    "block": ["Block", "Assessment_Unit"],
    "stage": ["Stage_of_Ground_Water_Extraction_pct", "Category"],
    "total extraction": ["Extraction_ham"],
    "annual recharge": ["Annual_Recharge_ham"],
    "recharge": ["Annual_Recharge_ham"],
    "extraction": ["Extraction_ham"],
    "availability": ["Groundwater_Level_m"],
    "water table": ["Groundwater_Level_m"],
    "level": ["Groundwater_Level_m"],
    "category": ["Category"],
}

def normalize_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).strip().lower()
    return re.sub(r"\s+", " ", s)

def find_metric_column(requested: str, columns: List[str]) -> Optional[str]:
    if not requested:
        return None
    req_norm = normalize_name(requested)

    # Priority 1: Exact alias matches
    for alias, targets in ALIAS_HINTS.items():
        if alias in req_norm:
            for t in targets:
                if t and t in columns:
                    return t

    # Priority 2: Direct column name matches
    for c in columns:
        if normalize_name(c) == req_norm:
            return c
    
    # Priority 3: Substring matches
    for c in columns:
        if req_norm in normalize_name(c):
            return c
    
    # Priority 4: Fuzzy matching
    match = difflib.get_close_matches(requested, columns, n=1, cutoff=0.5)
    return match[0] if match else None

# --------------- GEMINI WRAPPER ---------------

class LLM:
    def __init__(self, api_key: str, model: str):
        self.enabled = bool(api_key) and genai is not None
        self.model = model
        if self.enabled:
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
                print("[OK] Gemini API key is valid")
            except Exception as e:
                print(f"[WARN] Gemini initialization failed: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            print("[INFO] Running in fallback mode without AI")

    def classify(self, message: str, schema_cols: List[str]) -> Dict[str, Any]:
        if not self.enabled:
            return self._heuristic_classify(message)

        try:
            sys_prompt = """You are a groundwater data analysis assistant. Extract intent and structure from queries.
            Return JSON with: intent, metric, filters (state,district,block,unit,year), top_k.
            Intents: trend, forecast, compare, distribution, metric_lookup, help."""
            
            user_prompt = f"""Columns available: {schema_cols[:20]}
            User query: "{message}"
            Return only valid JSON:"""

            response = self.client.generate_content(
                f"{sys_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )
            
            content = response.text.strip()
            # Clean JSON response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip() if len(content.split('```')) > 2 else content
            
            data = json.loads(content)
            data.setdefault("filters", {})
            data.setdefault("top_k", 10)
            return data
            
        except Exception as e:
            print(f"[WARN] Gemini classification failed: {e}")
            return self._heuristic_classify(message)

    def _heuristic_classify(self, message: str) -> Dict[str, Any]:
        msg = message.lower().strip()
        if not msg:
            return {"intent": "help", "metric": None, "filters": {}, "top_k": 10}
        
        # Extract year
        year_match = re.search(r"\b(19|20)\d{2}\b", msg)
        year = int(year_match.group(0)) if year_match else None
        
        filters = {"year": year}
        
        # Extract geographic entities
        geo_patterns = {
            "state": r"(?:state|states|in|for)\s+([a-zA-Z][a-zA-Z\s]{2,})(?=\s|$)",
            "district": r"(?:district|districts|in|for)\s+([a-zA-Z][a-zA-Z\s]{2,})(?=\s|$)",
            "block": r"(?:block|blocks|assessment unit|unit)\s+([a-zA-Z][a-zA-Z\s]{2,})(?=\s|$)"
        }
        
        for geo_key, pattern in geo_patterns.items():
            match = re.search(pattern, msg, re.IGNORECASE)
            if match and match.group(1):
                filters[geo_key] = match.group(1).strip()
        
        # Enhanced intent detection
        intent_keywords = {
            "trend": ["trend", "over time", "history", "since", "yearly", "temporal"],
            "forecast": ["forecast", "predict", "next year", "future", "projection"],
            "compare": ["compare", "vs", "versus", "top", "highest", "lowest", "rank"],
            "distribution": ["distribution", "breakdown", "share", "percentage", "category", "pie"],
            "help": ["help", "what can", "how to", "show me", "example"]
        }
        
        intent = "metric_lookup"
        max_score = 0
        
        for intent_type, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in msg)
            if score > max_score:
                intent = intent_type
                max_score = score
        
        # Enhanced metric detection
        metric = None
        metric_keywords = {
            "Annual_Recharge_ham": ["recharge", "water recharge", "annual recharge"],
            "Extraction_ham": ["extraction", "water extraction", "total extraction"],
            "Groundwater_Level_m": ["groundwater", "water level", "water table", "depth", "level"],
            "Stage_of_Ground_Water_Extraction_pct": ["stage", "category", "extraction stage"],
            "Category": ["category", "stage"]
        }
        
        for metric_col, keywords in metric_keywords.items():
            if any(keyword in msg for keyword in keywords):
                metric = metric_col
                break
        
        # Extract top_k if specified
        top_k_match = re.search(r"top\s+(\d+)", msg)
        top_k = int(top_k_match.group(1)) if top_k_match else 10
        
        return {
            "intent": intent,
            "metric": metric,
            "filters": {k: v for k, v in filters.items() if v},
            "top_k": top_k,
        }

    def narrate(self, prompt: str) -> str:
        if not self.enabled:
            return self._simple_narrate(prompt)
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=400,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Gemini narration failed: {e}")
            return self._simple_narrate(prompt)

    def _simple_narrate(self, prompt: str) -> str:
        lines = prompt.split('\n')
        intent = "analysis"
        metric = "groundwater data"
        scope = "the selected region"
        forecast_info = ""
        
        for line in lines:
            line = line.strip()
            if "Intent:" in line:
                intent = line.split("Intent:")[1].strip().lower()
            elif "Metric:" in line:
                metric_part = line.split("Metric:")[1].strip()
                metric = re.sub(r'[_]', ' ', metric_part).replace('ham', '').replace('mm', '').title().strip()
            elif "Scope:" in line:
                scope = line.split("Scope:")[1].strip()
            elif "Forecast:" in line:
                forecast_info = line.split("Forecast:")[1].strip()
        
        narratives = {
            "trend": f"This analysis shows the historical trend of {metric} for {scope}.",
            "forecast": f"Based on historical data patterns, this forecast predicts future {metric} for {scope}.",
            "compare": f"This comparative analysis displays {metric} across different regions in {scope}.",
            "distribution": f"The distribution analysis shows how groundwater resources are categorized across different stages in {scope}.",
            "metric_lookup": f"This analysis examines {metric} data for {scope}."
        }
        
        return narratives.get(intent, f"This analysis provides insights into {metric} for {scope}.")

# Initialize LLM
LLM_CLIENT = LLM(GEMINI_API_KEY, GEMINI_MODEL)

# --------------- ANALYTICS CORE ---------------

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    
    for key in ["state", "district", "block", "unit"]:
        col = GEO.get(key)
        if col and filters.get(key):
            val = str(filters[key]).strip().lower()
            out = out[out[col].astype(str).str.lower().str.contains(val, na=False)]
    
    if YEAR_COL and filters.get("year") is not None:
        try:
            y = int(filters["year"])
            out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors='coerce')
            out = out[out[YEAR_COL] == y]
        except Exception:
            pass
    
    return out

def aggregate_by_year(df: pd.DataFrame, metric_col: str, agg: str = "mean") -> pd.DataFrame:
    if YEAR_COL not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()
    
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
    
    df = df[df[YEAR_COL].notna() & df[metric_col].notna()]
    
    if df.empty:
        return pd.DataFrame()
    
    grp = df.groupby(YEAR_COL)[metric_col]
    if agg == "mean":
        s = grp.mean()
    elif agg == "max":
        s = grp.max()
    elif agg == "min":
        s = grp.min()
    elif agg == "count":
        s = grp.count()
    else:
        s = grp.sum()
    
    return s.reset_index().sort_values(YEAR_COL)

def compare_top(df: pd.DataFrame, metric_col: str, level: str, year: Optional[int], top_k: int) -> pd.DataFrame:
    col = GEO.get(level)
    if not col or metric_col not in df.columns:
        return pd.DataFrame()
    
    working_df = df.copy()
    if year is not None and YEAR_COL in df.columns:
        working_df[YEAR_COL] = pd.to_numeric(working_df[YEAR_COL], errors='coerce')
        working_df = working_df[working_df[YEAR_COL] == year]
    
    working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors='coerce')
    working_df = working_df[working_df[metric_col].notna()]
    
    if working_df.empty:
        return pd.DataFrame()
    
    agg = working_df.groupby(col)[metric_col].mean().reset_index()
    agg = agg.sort_values(metric_col, ascending=False).head(top_k)
    agg.columns = [level.capitalize(), metric_col]
    
    return agg

def distribution_stage(df: pd.DataFrame) -> pd.DataFrame:
    if STAGE_COL not in df.columns:
        return pd.DataFrame()
    
    counts = df[STAGE_COL].replace(["", "nan", "None"], pd.NA).dropna().value_counts().reset_index()
    counts.columns = ["Stage", "Count"]
    return counts.sort_values("Count", ascending=False)

def ensure_timeseries_for_forecast(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if YEAR_COL not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce')
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    df = df[df[YEAR_COL].notna() & df[metric].notna()]
    
    if df.empty:
        return pd.DataFrame()
    
    ts = df.groupby(YEAR_COL)[metric].mean().reset_index()
    return ts.sort_values(YEAR_COL)

def forecast_next_year(ts_df: pd.DataFrame, metric: str) -> Tuple[int, float]:
    if ts_df.empty or len(ts_df) < 2:
        if not ts_df.empty:
            last_year = int(ts_df[YEAR_COL].iloc[-1])
            last_value = float(ts_df[metric].iloc[-1])
            return last_year + 1, last_value
        return datetime.now().year + 1, 0.0
    
    years = ts_df[YEAR_COL].astype(int).tolist()
    values = ts_df[metric].astype(float).to_numpy()
    
    if len(values) < 3:
        next_year = max(years) + 1
        forecast_value = np.mean(values[-2:]) if len(values) >= 2 else values[-1]
        return next_year, float(forecast_value)
    
    try:
        model = ExponentialSmoothing(values, trend="add", seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(1)[0]
        next_year = max(years) + 1
        return next_year, float(max(forecast, 0))
    except Exception:
        next_year = max(years) + 1
        if len(values) >= 2:
            slope = (values[-1] - values[-2]) / 1.0
            forecast_value = values[-1] + slope
        else:
            forecast_value = values[-1]
        return next_year, float(max(forecast_value, 0))

# --------------- CHART BUILDERS ---------------

def line_chart_from_ts(ts: pd.DataFrame, label: str, y_key: str) -> ChartSpec:
    labels = ts[YEAR_COL].astype(int).astype(str).tolist()
    data = ts[y_key].astype(float).round(2).tolist()
    
    return ChartSpec(
        type="line",
        data={
            "labels": labels, 
            "datasets": [{
                "label": label,
                "data": data,
                "borderColor": "rgb(59, 130, 246)",
                "backgroundColor": "rgba(59, 130, 246, 0.1)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.4,
            }]
        },
        options={
            "responsive": True,
            "maintainAspectRatio": False,
        }
    )

def bar_chart_from_df(x_vals: List[str], y_vals: List[float], label: str) -> ChartSpec:
    return ChartSpec(
        type="bar",
        data={
            "labels": x_vals, 
            "datasets": [{
                "label": label,
                "data": [round(float(v), 2) for v in y_vals],
                "backgroundColor": "rgba(59, 130, 246, 0.8)",
            }]
        },
        options={
            "responsive": True,
            "maintainAspectRatio": False,
        }
    )

def pie_chart_from_df(labels: List[str], vals: List[float]) -> ChartSpec:
    return ChartSpec(
        type="pie",
        data={
            "labels": labels,
            "datasets": [{
                "data": [float(v) for v in vals],
                "backgroundColor": [
                    "rgba(59, 130, 246, 0.8)",
                    "rgba(14, 165, 233, 0.8)",
                    "rgba(139, 92, 246, 0.8)",
                ],
            }]
        },
        options={
            "responsive": True,
            "maintainAspectRatio": False,
        }
    )

# --------------- INTENT ROUTER ---------------

def handle_chat(message: str, top_k: int, forced_year: Optional[int], debug: bool) -> ChatResponse:
    if not message.strip():
        return ChatResponse(
            intent="help",
            narrative="🚀 Welcome to Jal Mitra! Ask about groundwater data like:\n• 'Groundwater in Mumbai 2022'\n• 'Recharge trend Jaipur'\n• 'Compare districts by extraction'",
            metric=None,
            filters={}
        )
    
    schema_cols = list(DF.columns)
    classification = LLM_CLIENT.classify(message, schema_cols)

    intent = classification.get("intent", "metric_lookup")
    filters = classification.get("filters", {}) or {}
    metric_req = classification.get("metric")
    top_k_req = min(int(classification.get("top_k", top_k)), 20)
    
    if forced_year:
        filters["year"] = forced_year

    metric_col = None
    if intent != "distribution":
        metric_col = find_metric_column(metric_req or "", schema_cols)
        if metric_col is None and NUMERIC_COLS:
            metric_col = NUMERIC_COLS[0]

    dff = apply_filters(DF, filters)
    if dff.empty:
        return ChatResponse(
            intent=intent,
            metric=metric_col,
            filters=filters,
            narrative="❌ No data found. Try broadening your search.",
            chart=None,
            table=None
        )

    chart = None
    table = None
    forecast_payload = None

    try:
        if intent in ("trend", "forecast") and metric_col:
            ts = aggregate_by_year(dff, metric_col, "mean")
            if not ts.empty:
                chart = line_chart_from_ts(ts, label=metric_col, y_key=metric_col)
                if intent == "forecast":
                    ts2 = ensure_timeseries_for_forecast(dff, metric_col)
                    if not ts2.empty:
                        ny, val = forecast_next_year(ts2, metric_col)
                        forecast_payload = {"next_year": ny, "forecast_value": round(val, 2)}

        elif intent in ("compare", "top_n") and metric_col:
            for lvl in ["unit", "block", "district", "state"]:
                if GEO.get(lvl):
                    level = lvl
                    break
            else:
                level = "state"
            
            year = None
            if YEAR_COL in dff.columns and filters.get("year") is not None:
                try:
                    year = int(filters["year"])
                except (ValueError, TypeError):
                    year = None

            cmp_df = compare_top(dff, metric_col, level, year, top_k_req)
            if not cmp_df.empty:
                x = cmp_df[level.capitalize()].astype(str).tolist()
                y = cmp_df[metric_col].astype(float).tolist()
                chart = bar_chart_from_df(x, y, label=f"{metric_col} by {level.capitalize()}")
                table = cmp_df.to_dict(orient="records")

        elif intent == "distribution":
            dist_df = distribution_stage(dff)
            if not dist_df.empty:
                chart = pie_chart_from_df(
                    dist_df["Stage"].astype(str).tolist(), 
                    dist_df["Count"].astype(int).tolist()
                )
                table = dist_df.to_dict(orient="records")

        elif intent in ("metric_lookup", "help") and metric_col:
            if YEAR_COL in dff.columns and filters.get("year") is not None:
                for lvl in ["unit", "block", "district", "state"]:
                    col = GEO.get(lvl)
                    if col:
                        tmp = dff.groupby(col)[metric_col].mean().reset_index()
                        tmp = tmp.sort_values(metric_col, ascending=False).head(top_k_req)
                        if not tmp.empty:
                            table = tmp.to_dict(orient="records")
                            chart = bar_chart_from_df(
                                tmp[col].astype(str).tolist(),
                                tmp[metric_col].astype(float).tolist(),
                                label=f"{metric_col} ({filters.get('year')})"
                            )
                            break
            else:
                if YEAR_COL in dff.columns:
                    ts = aggregate_by_year(dff, metric_col, "mean")
                    if not ts.empty:
                        chart = line_chart_from_ts(ts, label=metric_col, y_key=metric_col)
                        table = ts.tail(10).to_dict(orient="records")
                else:
                    for lvl in ["unit", "block", "district", "state"]:
                        col = GEO.get(lvl)
                        if col:
                            tmp = dff.groupby(col)[metric_col].mean().reset_index()
                            tmp = tmp.sort_values(metric_col, ascending=False).head(top_k_req)
                            if not tmp.empty:
                                table = tmp.to_dict(orient="records")
                                chart = bar_chart_from_df(
                                    tmp[col].astype(str).tolist(),
                                    tmp[metric_col].astype(float).tolist(),
                                    label=f"{metric_col} by {lvl.capitalize()}"
                                )
                                break

    except Exception as e:
        print(f"[ERROR] Analytics processing failed: {e}")
        return ChatResponse(
            intent=intent,
            metric=metric_col,
            filters=filters,
            narrative=f"⚠️ Analysis error: {str(e)}",
            chart=None,
            table=None
        )

    fparts = []
    for key in ["state", "district", "block", "unit"]:
        if filters.get(key):
            fparts.append(f"{key.capitalize()}={filters[key]}")
    if filters.get("year"):
        fparts.append(f"Year={filters['year']}")
    
    scope = ", ".join(fparts) if fparts else "All India"
    
    nar_prompt = f"Analysis of {metric_col or 'groundwater data'} for {scope}. Intent: {intent}."
    if forecast_payload:
        nar_prompt += f" Forecast for {forecast_payload['next_year']}: {forecast_payload['forecast_value']:.2f}."
    
    narrative = LLM_CLIENT.narrate(nar_prompt)

    debug_info = None
    if debug:
        debug_info = {
            "classification_raw": classification,
            "resolved_metric": metric_col,
            "filters": filters
        }

    return ChatResponse(
        intent=intent,
        metric=metric_col,
        filters=filters,
        chart=chart,
        table=table,
        narrative=narrative,
        forecast=forecast_payload,
        debug_info=debug_info,
    )

# --------------- FASTAPI APP ---------------

app = FastAPI(title="Jal Mitra - Groundwater Analytics AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Jal Mitra API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "ok": True,
        "rows": int(DF.shape[0]),
        "cols": int(DF.shape[1]),
        "year_col": YEAR_COL,
        "geo_cols": GEO,
        "stage_col": STAGE_COL,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/schema")
async def schema():
    return {
        "columns": list(DF.columns),
        "numeric_columns": NUMERIC_COLS,
        "text_columns": TEXT_COLS,
        "year_column": YEAR_COL,
        "geo_columns": GEO,
        "stage_column": STAGE_COL,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return handle_chat(
            message=req.message,
            top_k=req.top_k,
            forced_year=req.year,
            debug=req.debug,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

