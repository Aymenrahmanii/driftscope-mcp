```markdown
# 🔬 DriftScope — Autonomous MLOps & Statistical Diagnostics MCP Server

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Protocol](https://img.shields.io/badge/MCP-Model_Context_Protocol-purple.svg?style=flat)](https://modelcontextprotocol.io/)
[![Deployment](https://img.shields.io/badge/Render-Live_Cloud_SSE-46E3B7.svg?style=flat&logo=render&logoColor=white)](https://driftscope-mcp.onrender.com)
[![Engine](https://img.shields.io/badge/Polars-Blazing_Fast_Data-CD792C.svg?style=flat&logo=polars&logoColor=white)](https://pola.rs)

> An open-source, cloud-deployed **Model Context Protocol (MCP)** server that equips Large Language Models (LLMs) with deterministic statistical computing engines to audit data drift and model degradation in production tabular pipelines.

---

## 💡 The Problem

LLMs are exceptional at high-level reasoning, code generation, and root-cause analysis, but notoriously unreliable at **precise statistical math**. When monitoring machine learning pipelines, asking an LLM to evaluate distribution shift directly leads to severe numerical hallucinations.

**DriftScope** solves this by bridging the LLM to a dedicated Python analytical engine via the open **Model Context Protocol**:
- **The LLM** handles orchestration, triage, hypothesis generation, and incident reporting.
- **DriftScope** executes deterministic, vector-accelerated hypothesis testing (**Two-sample Kolmogorov-Smirnov**) and **Population Stability Index (PSI)** calculations using **Polars** and **SciPy**.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                      LLM Host                          │
│         (Claude Desktop / Cursor / Custom Agent)       │
└───────────────────────────┬────────────────────────────┘
                            │ JSON-RPC 2.0
                            ▼ (mcp-remote bridge)
               ┌────────────────────────┐
               │    Internet / HTTPS    │
               └────────────┬───────────┘
                            │ Server-Sent Events (SSE)
                            ▼
┌────────────────────────────────────────────────────────┐
│             DriftScope MCP Server (Render)             │
│  ┌──────────────────────────────────────────────────┐  │
│  │                  FastMCP Router                  │  │
│  └────────┬───────────────────────┬───────────────┬─┘  │
│           │                       │               │    │
│           ▼                       ▼               ▼    │
│    [ Tools Engine ]      [ Resources Hub ]   [ Prompts ]│
│    • check_drift         • standards://      • audit_   │
│    • compute_psi           drift-policy        feature │
│    • generate_mock                                     │
│           │                                            │
│           ▼                                            │
│    [ Data Layer: Polars + SciPy + NumPy ]              │
└────────────────────────────────────────────────────────┘
```

---

## ✨ Features & MCP Primitives

### 1. 🛠️ Tools (Callable Actions)
- `check_feature_drift(baseline_csv, current_csv, feature_column, significance_level)`: Executes two-sample **Kolmogorov-Smirnov (KS)** tests to detect continuous covariate shift with rigorous $p$-values.
- `compute_psi(baseline_csv, current_csv, feature_column, bins)`: Computes the **Population Stability Index (PSI)** with equal-frequency baseline binning and Laplace smoothing to classify shift severity (`Stable`, `Moderate`, or `Critical`).
- `generate_mock_datasets()`: Synthesizes baseline and drifted production samples on the fly for pipeline verification.

*Supports both local cloud file paths and remote public HTTP/HTTPS URLs (AWS S3, GitHub raw, etc.).*

### 2. 📂 Resources (Passive Knowledge)
- `standards://drift-policy`: Exposes organization-wide MLOps threshold standards (e.g., $p < 0.05$ rejection criteria, PSI warning zones) directly into the agent's context.

### 3. 📝 Prompts (Reusable Workflows)
- `audit_feature`: Pre-engineered diagnostic prompt guiding the model through end-to-end drift triage, impact evaluation, and retraining recommendations.

---

## 🚀 Live Cloud Deployment

DriftScope is deployed as a live cloud service on **Render** utilizing the **Server-Sent Events (SSE)** transport.

- **Live SSE Endpoint**: `https://driftscope-mcp.onrender.com/sse`

---

## 🔌 Quickstart: Connect to Claude Desktop

You can connect your local Claude Desktop to the live cloud deployment in seconds:

1. Open your Claude Desktop configuration file:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add `driftscope-cloud` to your `mcpServers` object:

```json
{
  "mcpServers": {
    "driftscope-cloud": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://driftscope-mcp.onrender.com/sse"
      ]
    }
  }
}
```

3. Restart Claude Desktop. The 🔨 **hammer icon** will appear in the chat interface showing your active tools!

---

## 📊 Sample Interaction & Output

### Prompt:
> *"Audit both the 'income' and 'age' features for drift using DriftScope and give me an MLOps summary."*

### Output:
```text
## MLOps Drift Summary

| Feature | KS Stat | p-value | PSI Score | Verdict |
|---|---|---|---|---|
| income  | 0.3960  | 0.00000 | 0.7060    | 🔴 Critical shift |
| age     | 0.0840  | 0.05910 | 0.0230    | 🟢 Stable |

Attention needed: income
- Both the KS test and PSI confirm income has drifted severely (p-value ≈ 0, PSI = 0.706 > 0.20 threshold).
- Recommendation: Upstream investigation required; trigger model retraining fallback pipeline.
```

---

## 💻 Local Development

Clone the repository and run locally using `uv`:

```bash
git clone https://github.com/Aymenrahmanii/driftscope-mcp.git
cd driftscope-mcp

# Install dependencies
uv sync

# Launch the interactive MCP Inspector UI
uv run mcp dev server.py
```

---

## 📦 Tech Stack

- **Protocol**: Model Context Protocol (MCP) Python SDK
- **Framework**: FastMCP (Starlette, Uvicorn, SSE)
- **Data & Math**: Polars, NumPy, SciPy, Scikit-learn, HTTPX
- **Infrastructure**: Render Web Services, GitHub
```