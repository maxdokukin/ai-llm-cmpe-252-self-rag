# Agentic Self-RAG — three extensions to Self-RAG: prompt engineering, semantic routing, agentic verification

CMPE 252 — Artificial Intelligence and Data Engineering · Fall 2025 · Team: Ayushi Bhatnagar, Maxim Dokukin, Krushna Thakkar · Completed

## Overview

The project reproduces the Self-RAG framework (Asai et al., ICLR 2024) with the public `selfrag_llama2_7b` model, a
Contriever/FAISS retriever and a Wikipedia subset, then extends it along three axes. A prompt-engineering layer adds a
retrieval-decision classifier, a strict grounding template with citations and an "I don't know" fallback, a targeted
self-critique step and a grounding-first LLM judge. A semantic router classifies each query as SIMPLE, RAG or TOOL and
dispatches it to direct generation, the Self-RAG pipeline or deterministic Python tools. An agentic layer runs the
Self-RAG checks (Retrieve? IsRelevant? IsSupported? IsUseful?) at every step of a multi-step plan and records
JSON-structured traces. Each extension is a self-contained Colab notebook with its own A/B evaluation against the
unrouted, unprompted baseline.

## Highlights

- Semantic router: 44.7% lower average latency (2.27 s → 1.26 s) and +0.27 judged quality (2.30 → 2.57 on a 1–5 scale) on a 30-query A/B test; tool-routed queries dropped from 1.73 s to 0.08 s (`src/model_routing/tests/router_vs_rag_ab_test_20251208_233903.csv`).
- Prompt engineering: grounding-first judge scores rose from 0 to 5 on seven of eight evaluation queries (mean 0.25 → 4.5) without touching model weights (`src/prompt_engineering`, cell 12).
- Agentic layer: reliability score 0.48 → 0.73 across seven reasoning scenarios covering hallucination resistance, tool-use decisions and long-horizon planning (report §VI).
- Deterministic tools removed arithmetic and counting hallucinations (baseline answered "14" for a three-word count; tool route answered 3).

## How it works

```
query → Router (few-shot intent prompt on the Self-RAG LM)
      ├─ SIMPLE → direct generation, minimal system prompt
      ├─ RAG    → retrieve top-k (Contriever + FAISS) → grounded prompt → generate → self-critique → judge
      └─ TOOL   → regex/NL-to-expression parsing → Python calculator / word counter
agentic mode: plan step → Retrieve? → IsRelevant? → IsSupported? → IsUseful? → revise → next step (JSON trace)
```

- **Baseline (`src/initial_contribution`)** — Self-RAG 7B served with vLLM on a Colab A100; retrieval only when the model emits `[Retrieval]`; utility-token selection among per-passage candidates; custom qa / explanatory / chain-of-thought / compare-contrast prompt modes and a reflection pass.
- **Prompt engineering (`src/prompt_engineering`)** — `build_prompt` (evidence summary + final answer + `[TITLE]` citation + fallback), `critique_prompt` (edit only the wrong parts), `judge_prompt` (JSON, grounding weighted first, regex fallback extraction).
- **Semantic router (`src/model_routing`)** — `route_query` few-shot classifier with negative constraints ("GPT-3" is RAG, not TOOL), `smart_ask` dispatcher, `clean_artifacts` to strip `[Utility:n]` / `<paragraph>` tags, 30-query A/B loop and LLM-judge scoring.
- **Agentic layer (`src/agents`)** — step-wise planner over the Self-RAG loop, heuristic reliability scoring (0.4 hallucination + 0.3 evidence consistency + 0.3 step verification), bar-chart and heatmap comparison.

## Results

| Experiment | Metric | Baseline | Extended | Note |
|---|---|---|---|---|
| Router A/B (N = 30) | avg latency | 2.272 s | 1.256 s | −44.7 % |
| Router A/B (N = 30) | avg quality, LLM judge 1–5 | 2.30 | 2.57 | +0.27 |
| Router, TOOL queries (12) | avg latency | 1.728 s | 0.075 s | −95 % |
| Router, SIMPLE queries (7) | avg latency | 2.440 s | 0.813 s | −66 % |
| Router, RAG queries (11) | avg latency | 2.758 s | 2.826 s | +0.06 s router overhead |
| Prompt engineering (8 queries) | grounding-first judge 0–5 | 0.25 mean | 4.5 mean | 7/8 queries scored 5 |
| Agentic layer (7 scenarios) | reliability 0–1 | 0.48 | 0.73 | heuristic composite |

The router gains come from skipping retrieval on conversational queries and replacing generation with Python on
computational ones; RAG-routed queries pay a small classification overhead. A second A/B run on the same day
(`…_233308.csv`) reproduced the latency gain (2.29 s → 1.29 s) but showed a smaller quality gain (+0.07), so the
quality delta should be read as indicative rather than precise.

## Getting started

```bash
# Google Colab, A100 recommended (7B model via vLLM + ~9 GB Wikipedia demo index)
pip install vllm faiss-cpu transformers sentence-transformers accelerate gdown
# open one notebook and run cells top to bottom:
#   src/initial_contribution/Initial Contribution.ipynb                          cells 0–12  baseline + reflection
#   src/prompt_engineering/PromptEngineeringSelfRag_CMPE_252_RAG_FINAL.ipynb      cells 0–12  grounding template + judge
#   src/model_routing/model_router.ipynb                                          cells 0–20  router A/B test (writes CSV)
#   src/agents/Agentic_Appraoch-2.ipynb                                           cells 0–5   agentic benchmark + charts
```

Cells 0–4 of each notebook verify the GPU, install dependencies, clone the upstream Self-RAG repository, download the
Wikipedia subset and load `selfrag_llama2_7b`; a Hugging Face token is not required.

## Documents

- [Final report](doc/report_copy.pdf) — Bhatnagar, Dokukin, Thakkar, 6 pp.
- [Final slides](doc/final_slides.pdf) — 14 slides
- [Midterm slides](src/initial_contribution/Midsem%20Presentation.pdf) — paper overview and initial contribution
- Demo videos: links in each `src/*/video.txt`
- Router A/B logs: `src/model_routing/tests/*.csv`; router figures: `src/model_routing/figures/`
- Upstream paper: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ICLR 2024
