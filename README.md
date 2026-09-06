---

### Part 2: What to Record for Your Demo (30–45s Video/GIF)

A short, visually clear demo will get 10x more engagement on LinkedIn and impress technical interviewers. 

#### Recommended Tool:
- **Windows**: Download **[ScreenToGif](https://www.screentogif.com/)** (free, lightweight, directly outputs high-quality `.gif` or `.mp4`).

---

#### The 40-Second Storyboard:

| Time | Screen Area | What to Show |
| :--- | :--- | :--- |
| **0:00 – 0:05** | Claude Desktop | Click the 🔨 **hammer icon** in Claude Desktop to briefly show `driftscope-cloud` and its tools (`check_feature_drift`, `compute_psi`, `generate_mock_datasets`). |
| **0:05 – 0:12** | Chat prompt | Paste this prompt:<br>`"Generate mock data and check 'income' and 'age' for drift using KS-test and PSI."` and press Send. |
| **0:12 – 0:25** | Chat stream | Show Claude calling the tools in succession. The tool approval boxes and spinners show real client-server communication. |
| **0:25 – 0:35** | Chat stream | Claude outputs the formatted **MLOps Drift Summary table** with the red/green indicators ($\text{PSI} = 0.706$ vs $0.023$). |
| **0:35 – 0:42** | *(Optional Split screen or Tab switch)* | Switch to your browser showing the **Render Live Logs** with `POST /messages/ ... CallToolRequest 202 Accepted` to prove it is executing live in the cloud. |

---

### Part 3: Ready-to-Post LinkedIn Caption

When you post your recording on LinkedIn, use this copy:

> **I built and deployed a cloud-native Model Context Protocol (MCP) server: DriftScope 🔬**
>
> LLMs are great at reasoning, but notoriously prone to numerical hallucinations when calculating statistical distributions. 
>
> To solve this, I created **DriftScope** — an MLOps diagnostics engine that connects Claude Desktop directly to a remote Python analytical server over Server-Sent Events (SSE).
>
> 🚀 **How it works:**
> 1. Claude orchestrates the workflow and identifies pipeline anomalies.
> 2. DriftScope runs deterministic **Two-Sample Kolmogorov-Smirnov tests** and computes **Population Stability Index (PSI)** using **Polars** and **SciPy**.
> 3. Deployed live on **Render** using FastMCP and modern Python tooling (`uv`).
>
> 📂 GitHub Repo: https://github.com/Aymenrahmanii/driftscope-mcp
>
> #MachineLearning #MLOps #DataScience #Python #AI #ModelContextProtocol #Claude #Polars