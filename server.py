import os
import io
import httpx
import numpy as np
import polars as pl
from scipy import stats
from mcp.server.fastmcp import FastMCP

# Render provides the port in the PORT environment variable (defaults to 8000 locally)
PORT = int(os.environ.get("PORT", 8000))

# Initialize FastMCP bound to 0.0.0.0 for external cloud access
mcp = FastMCP("DriftScope-MLOps", host="0.0.0.0", port=PORT)


def load_dataset(path_or_url: str) -> pl.DataFrame:
    """Helper to read CSV from either a local file or a remote HTTP(S) URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = httpx.get(path_or_url, timeout=15.0)
        resp.raise_for_status()
        return pl.read_csv(io.BytesIO(resp.content))
    return pl.read_csv(path_or_url)


# ==========================================
# 1. TOOLS
# ==========================================

@mcp.tool()
def generate_mock_datasets() -> dict:
    """Generates synthetic baseline and production CSVs in the local cloud storage."""
    np.random.seed(42)
    n = 500

    age_base = np.random.normal(35, 10, n)
    income_base = np.random.normal(50000, 15000, n)

    age_curr = np.random.normal(35.5, 9.8, n)  # Stable
    income_curr = np.random.normal(65000, 18000, n)  # Drifted!

    os.makedirs("data", exist_ok=True)
    pl.DataFrame({"age": age_base, "income": income_base}).write_csv("data/baseline.csv")
    pl.DataFrame({"age": age_curr, "income": income_curr}).write_csv("data/current.csv")

    return {
        "status": "Mock datasets generated successfully in ./data folder!",
        "baseline_path": "data/baseline.csv",
        "current_path": "data/current.csv",
        "features": ["age", "income"],
    }


@mcp.tool()
def check_feature_drift(
    baseline_csv: str,
    current_csv: str,
    feature_column: str,
    significance_level: float = 0.05
) -> dict:
    """
    Evaluates covariate shift for a continuous feature between baseline and production
    datasets using the Kolmogorov-Smirnov (KS) test.
    Accepts both local file paths and public HTTP/HTTPS URLs.
    """
    try:
        df_base = load_dataset(baseline_csv)
        df_curr = load_dataset(current_csv)

        if feature_column not in df_base.columns or feature_column not in df_curr.columns:
            return {"error": f"Column '{feature_column}' not found in both datasets."}

        val_base = df_base[feature_column].drop_nulls().to_numpy()
        val_curr = df_curr[feature_column].drop_nulls().to_numpy()

        ks_stat, p_value = stats.ks_2samp(val_base, val_curr)
        is_drifted = bool(p_value < significance_level)

        return {
            "feature": feature_column,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 5),
            "is_drift_detected": is_drifted,
            "status": "ALERT: Significant distribution shift detected!" if is_drifted else "STABLE: Distribution matches baseline."
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def compute_psi(
    baseline_csv: str,
    current_csv: str,
    feature_column: str,
    bins: int = 10
) -> dict:
    """
    Computes Population Stability Index (PSI) to evaluate feature shift severity.
    Accepts both local file paths and public HTTP/HTTPS URLs.
    """
    try:
        df_base = load_dataset(baseline_csv)[feature_column].drop_nulls().to_numpy()
        df_curr = load_dataset(current_csv)[feature_column].drop_nulls().to_numpy()

        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(df_base, quantiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        base_counts, _ = np.histogram(df_base, bins=bin_edges)
        curr_counts, _ = np.histogram(df_curr, bins=bin_edges)

        base_pct = (base_counts + 1e-4) / len(df_base)
        curr_pct = (curr_counts + 1e-4) / len(df_curr)

        psi_val = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))

        severity = "Stable (PSI < 0.1)"
        if psi_val >= 0.2:
            severity = "Critical Shift (PSI >= 0.2): Retraining recommended."
        elif psi_val >= 0.1:
            severity = "Moderate Shift (0.1 <= PSI < 0.2): Monitor closely."

        return {
            "feature": feature_column,
            "psi_score": round(psi_val, 4),
            "interpretation": severity
        }
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 2. RESOURCES
# ==========================================

@mcp.resource("standards://drift-policy")
def get_drift_policy() -> str:
    """Returns organization monitoring standards."""
    return """
    # MLOps Monitoring Thresholds Policy:
    - KS-Test: Reject null hypothesis if p-value < 0.05 (statistically significant shift).
    - PSI:
      * < 0.10: Stable, no intervention.
      * 0.10 - 0.20: Warning zone, monitor closely.
      * > 0.20: Critical drift, automatic fallback or retraining trigger.
    """


# ==========================================
# 3. PROMPTS
# ==========================================

@mcp.prompt()
def audit_feature(feature_name: str) -> str:
    """Guides the model through a full statistical drift triage."""
    return f"""
    Please audit the feature '{feature_name}':
    1. Run both the KS-Test and PSI tools on data/baseline.csv and data/current.csv.
    2. Read 'standards://drift-policy' to interpret the score.
    3. State clearly whether the model needs retraining.
    """


# ==========================================
# 4. SERVER RUNNER (SSE)
# ==========================================

if __name__ == "__main__":
    # Runs the Server-Sent Events (SSE) server over HTTP
    mcp.run(transport="sse")