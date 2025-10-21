"""
run.py — IPC/CPC Automatic Inference + Local Validation (Resumable, No Duplicates, Auto-batching)

Features:
- Longer, more stable System Prompt (bilingual Chinese/English, JSON-only)
- Parameterization: directories/column names/thresholds/model/temperature/batch size, etc. (no globals)
- Automatic batch size estimation (--auto-batch; based on context limit + average sample length)
- LLM returns JSON-only by default; falls back to "extract JSON from text" on failure
- Result caching (hashes each batch payload; skips LLM call on hit)
- Deterministic naming for raw inference files (avoids duplicate copies)
- Checkpoint-based resumable execution (skips completed sub-categories; checkpoints can be manually deleted to re-run)
- Wildcard matching (supports * at the end of expected codes)
- NaN cleaning, safe filenames (Windows-friendly), unified logging
"""

from __future__ import annotations
import os, re, json, time, uuid, math, hashlib, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ========= Initialize OpenAI (client created later) =========
load_dotenv()
from openai import OpenAI

# ========= Extended System Prompt (Bilingual Chinese/English) =========
SYSTEM_PROMPT_INFER = """
You are "IPC/CPC Code Inference", a careful bilingual (中文/English) classifier.

## Core Objective
Given batches of patents (title + abstract), infer up to 5 **most likely** IPC/CPC codes **per patent**.

## Strict Rules
1) No validation against any external mapping – only infer from title/abstract + taxonomy knowledge.
2) Normalization:
   - Uppercase, remove spaces; keep slashes. Example: "G01S 17/894" → "G01S17/894".
   - Deduplicate codes.
3) Specificity preference:
   - If evidence is clear, prefer **subgroup** (e.g., G01S17/894).
   - Otherwise fall back to **main group** (e.g., G01S17/89).
4) Conservativeness:
   - If uncertain, output fewer codes (≤5). Avoid hallucinations and irrelevant sections.
5) Language Agnostic:
   - Titles/abstracts may be Chinese or English; analyze both reliably.

## Output Format (JSON ONLY, no extra text / no markdown)
{
  "patent_batches": [
    {
      "domain": "…",
      "tech_name_raw": "…",
      "patents": [
        {"patent_id": "…", "inferred_codes": ["CODE1","CODE2", "..."]}
      ]
    }
  ]
}

Key points of the Chinese prompt (Do not output this section, execute according to these requirements):
- Infer only based on title/abstract, without referencing any lookup tables.
- Codes should be uppercase, no spaces, slashes retained; maximum of 5; provide subgroups when possible.
- When uncertain, provide fewer codes.
""".strip()

# ========= Local Output Schema (retained for extension, not currently enforced) =========
OUTPUT_SCHEMA = {
  "name": "tech_validation_result",
  "schema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "object",
        "properties": {
          "domains": { "type": "array", "items": { "type": "string" } },
          "techs_total": { "type": "integer" },
          "supported": { "type": "integer" },
          "partially_supported": { "type": "integer" },
          "not_supported": { "type": "integer" }
        },
        "required": ["domains","techs_total","supported","partially_supported","not_supported"]
      },
      "tech_results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "domain": { "type": "string" },
            "tech_name_expected": { "type": "string" },
            "tech_name_resolved_from": { "type": "string" },
            "tech_resolution": { "type": "string", "enum": ["exact","alias","token","unmapped"] },
            "expected_codes": { "type": "array", "items": { "type": "string" } },
            "stats": {
              "type": "object",
              "properties": {
                "n_patents": { "type": "integer" },
                "rate_A": { "type": "number" },
                "rate_B": { "type": "number" },
                "rate_C": { "type": "number" },
                "rate_none": { "type": "number" }
              },
              "required": ["n_patents","rate_A","rate_B","rate_C","rate_none"]
            },
            "verdict": { "type": "string", "enum": ["supported","partially_supported","not_supported"] },
            "confidence": { "type": "number" },
            "evidence_samples": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "patent_id": { "type": "string" },
                  "code_match_level": { "type": "string", "enum": ["A","B","C","none"] },
                  "inferred_codes": { "type": "array", "items": { "type": "string" } },
                  "reasoning": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["patent_id","code_match_level","reasoning"]
              }
            }
          },
          "required": ["domain","tech_name_expected","expected_codes","stats","verdict","confidence"]
        }
      }
    },
    "required": ["summary","tech_results"]
  }
}

# ========= Configuration =========
@dataclass
class Config:
    mappings_dir: Path
    patents_dir: Path
    outputs_dir: Path
    techset_path: Path

    # CSV Columns
    col_id: str = "patent id"
    col_title: str = "title"
    col_abs: str = "abstract"

    # Thresholds
    tau_strong: float = 0.35
    tau_min: float = 0.15

    # Number of evidence samples
    max_evidence: int = 5

    # Model
    model: str = "gpt-5"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int | None = None

    # Batching
    batch_size: int = 30
    auto_batch: bool = False
    context_limit: int = 128_000  # Token limit for auto-batching

    # Network timeout & proxy
    connect_timeout: float = 15.0
    read_timeout: float = 180.0
    proxy: str | None = os.getenv("HTTPS_PROXY") or os.getenv("ALL_PROXY")

    # Auto-batch cap (to avoid sudden spikes to 120+)
    auto_batch_cap: int = 60

# ========= Basic Utilities =========
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_outputs(cfg: Config) -> None:
    (cfg.outputs_dir / "raw").mkdir(parents=True, exist_ok=True)
    (cfg.outputs_dir / "summary").mkdir(parents=True, exist_ok=True)
    (cfg.outputs_dir / "state").mkdir(parents=True, exist_ok=True)

def _safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '_', str(s))

def _norm(code: str) -> str:
    if not isinstance(code, str): return ""
    return code.upper().replace(" ", "").strip()

def _main_group(code: str) -> str:
    c = _norm(code)
    return c.split("/")[0] if "/" in c else c

STAR = "*"
def _normalize_pattern(ec: str) -> tuple[str,bool]:
    ec = _norm(ec)
    return (ec[:-1], True) if ec.endswith(STAR) else (ec, False)

def _match_level_one(ic: str, expected_codes: List[str]) -> str:
    """
    A: Exact/prefix/wildcard(*); B: Main group match; none: Other
    """
    icn = _norm(ic)
    if not icn: return "none"
    exps = [_norm(x) for x in expected_codes]

    # A
    for ec_raw in exps:
        ec, star = _normalize_pattern(ec_raw)
        if icn == ec:
            return "A"
        if "/" in ec:
            if star and icn.startswith(ec):
                return "A"
            if not star and _main_group(icn) == _main_group(ec) and icn.startswith(ec):
                return "A"
    # B
    icg = _main_group(icn)
    for ec_raw in exps:
        if icg and icg == _main_group(ec_raw):
            return "B"
    return "none"

def chunk_list(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ========= Automatic Batch Size Estimation =========
def _rough_tokens_of_text(s: str) -> int:
    if not s: return 0
    return max(1, int(len(s) / 3.5))  # Rough estimate: 3.5 chars ≈ 1 token

def _rough_tokens_of_item(title: str, abstract: str) -> int:
    return _rough_tokens_of_text(title) + _rough_tokens_of_text(abstract) + 12  # JSON overhead

def choose_dynamic_batch_size(records: List[dict], cfg: Config) -> int:
    if not records: return 5
    sample = records[:50]
    toks = []
    for r in sample:
        t = str(r.get(cfg.col_title, ""))[:2000]
        a = str(r.get(cfg.col_abs, ""))[:6000]
        toks.append(_rough_tokens_of_item(t, a))
    avg_item = max(1, int(sum(toks)/len(toks)))
    # Budget: leave 45-55% of the limit as a safety margin
    safety_ratio = 0.55
    system_overhead = 800
    per_batch_overhead = 350
    budget = max(0, int(cfg.context_limit * safety_ratio) - (system_overhead + per_batch_overhead))
    if budget <= 0: return 5
    est = max(5, min(cfg.auto_batch_cap, int(budget / avg_item)))
    return est

# ========= Checkpoints (for true resumable execution) =========
def _ckpt_path(cfg: Config) -> Path:
    return cfg.outputs_dir / "state" / "checkpoints.json"

def _load_ckpt(cfg: Config) -> Dict[str, str]:
    p = _ckpt_path(cfg)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_ckpt(cfg: Config, d: Dict[str, str]) -> None:
    p = _ckpt_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

# ========= Mapping/Techset =========
def detect_mapping_file(cfg: Config, domain: str) -> Path | None:
    patterns = [f"{domain}.xlsx", f"{domain}.xls", f"{domain}.csv",
                f"{domain}*.xlsx", f"{domain}*.xls", f"{domain}*.csv"]
    for pat in patterns:
        hits = list(cfg.mappings_dir.glob(pat))
        if hits: return hits[0]
    return None

def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gb18030"]:
            try:
                try: return pd.read_csv(path, encoding=enc)
                except Exception: return pd.read_csv(path, encoding=enc, sep=";")
            except Exception: continue
        return pd.read_csv(path, encoding="utf-8")
    raise ValueError(f"Unsupported mapping file type: {path}")

def read_mapping_table_to_items(tab_path: Path) -> List[Dict[str, Any]]:
    df = _read_table(tab_path)
    cols = [c for c in df.columns if isinstance(c, str)]

    name_col = None
    # Keywords for name column: "name", "technology", "tech", "tech_name", "sub-category", "category"
    for kw in ["名称","技术","tech","技术名称","小类","类别","子类"]:
        name_col = next((c for c in cols if kw.lower() in c.lower()), None)
        if name_col: break
    if not name_col: name_col = cols[0]

    # Keywords for code columns: "code", "ipc", "cpc", "classification_code"
    code_cols = [c for c in cols if any(k in c.lower() for k in ["code","ipc","cpc","分类号","分类代码"])]
    if not code_cols: code_cols = [c for c in cols if c != name_col]

    items = []
    for _, row in df.iterrows():
        name = str(row.get(name_col,"")).strip()
        if not name or name.lower()=="nan": continue
        codes = []
        for cc in code_cols:
            v = row.get(cc, None)
            if pd.isna(v): continue
            # Split by: ,，;；/ |
            parts = re.split(r"[,\uFF0C;；/ \|]+", str(v))
            for p in parts:
                p = p.upper().strip().replace(" ", "")
                if p: codes.append(p)
        codes = sorted(list(dict.fromkeys(codes)))
        if codes: items.append({"tech_name": name, "codes": codes})
    return items

def load_expected_code_mapping(cfg: Config, domains: List[str]) -> List[Dict[str, Any]]:
    out = []
    for d in domains:
        path = detect_mapping_file(cfg, d)
        if not path or not path.exists():
            log(f"[WARN] Mapping file not found for domain: {d} (supports *.xlsx/*.xls/*.csv)")
            continue
        out.append({"domain": d, "items": read_mapping_table_to_items(path)})
    return out

def resolve_expected_codes(domain: str, tech_name_raw: str, expected_map: List[Dict[str, Any]]) -> List[str]:
    dm = next((x for x in expected_map if x.get("domain")==domain), None)
    if not dm: return []
    items = dm.get("items", [])
    for it in items:
        if str(it.get("tech_name","")).strip() == tech_name_raw.strip():
            return it.get("codes", [])
    for it in items:
        name = str(it.get("tech_name",""))
        if name and (name in tech_name_raw or tech_name_raw in name):
            return it.get("codes", [])
    return items[0].get("codes", []) if items else []

# ========= Read Patent CSVs =========
def read_patent_csvs_for_domain(cfg: Config, domain_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    pairs: List[Tuple[str,pd.DataFrame]] = []
    for p in sorted(domain_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="gb18030")
        tech_name_raw = p.stem

        for col in [cfg.col_title, cfg.col_abs]:
            if col not in df.columns:
                raise ValueError(f"{p.name} is missing column: {col}")

        df[cfg.col_title] = df[cfg.col_title].fillna("").astype(str)
        df[cfg.col_abs]   = df[cfg.col_abs].fillna("").astype(str)
        df = df[(df[cfg.col_title]!="") | (df[cfg.col_abs]!="")].copy()

        if cfg.col_id not in df.columns:
            df[cfg.col_id] = [f"{tech_name_raw}-{i}" for i in range(len(df))]

        pairs.append((tech_name_raw, df))
    return pairs

# ========= Validation Statistics =========
def validate_one_tech(cfg: Config,
                      domain: str,
                      tech_name_raw: str,
                      inferred_patents: List[Dict[str, Any]],
                      expected_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    exp_codes = resolve_expected_codes(domain, tech_name_raw, expected_map)
    n = len(inferred_patents)
    hitsA = hitsB = hitsNone = 0
    evidence = []

    for p in inferred_patents:
        inferred = p.get("inferred_codes", []) or []
        best = "none"
        for ic in inferred:
            lvl = _match_level_one(ic, exp_codes)
            if lvl == "A": best="A"; break
            elif lvl == "B": best="B"
        if best == "A": hitsA += 1
        elif best == "B": hitsB += 1
        else: hitsNone += 1

        if len(evidence) < cfg.max_evidence:
            evidence.append({
                "patent_id": str(p.get("patent_id","")),
                "code_match_level": best,
                "inferred_codes": inferred,
                "reasoning": ["Local validation: Scored based on exact/prefix/main-group relationship between inferred codes and the lookup table"]
            })

    rate_A = hitsA/n if n else 0.0
    rate_B = hitsB/n if n else 0.0
    rate_none = hitsNone/n if n else 0.0
    rate_C = 0.0

    ab = rate_A + rate_B
    if ab >= cfg.tau_strong: verdict="supported"
    elif ab >= cfg.tau_min:  verdict="partially_supported"
    else:                    verdict="not_supported"

    return {
        "domain": domain,
        "tech_name_expected": tech_name_raw,
        "tech_name_resolved_from": tech_name_raw,
        "tech_resolution": "token",
        "expected_codes": exp_codes,
        "stats": {
            "n_patents": n,
            "rate_A": round(rate_A,4),
            "rate_B": round(rate_B,4),
            "rate_C": round(rate_C,4),
            "rate_none": round(rate_none,4),
        },
        "verdict": verdict,
        "confidence": round(0.7 + 0.2*ab, 3),
        "evidence_samples": evidence
    }

# ========= LLM Call (with cache + fallback) =========
def _extract_first_json_object(text: str) -> str:
    if not text or not text.strip():
        raise json.JSONDecodeError("empty", "", 0)
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        json.loads(s); return s
    except Exception:
        pass
    stack = 0; start = -1
    for i,ch in enumerate(s):
        if ch == '{':
            if stack==0: start=i
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack==0 and start!=-1:
                cand = s[start:i+1]
                try:
                    json.loads(cand); return cand
                except Exception:
                    continue
    raise json.JSONDecodeError("no json object found", s[:200], 0)

def _payload_cache_key(obj: dict) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(b).hexdigest()

@retry(wait=wait_exponential(multiplier=1, min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, Exception)))
def call_llm_infer(cfg: Config, client: OpenAI, input_payload: dict) -> dict:
    """
    Returns: {"patent_batches":[{domain, tech_name_raw, patents:[{patent_id, inferred_codes:[]}, ...]}]}
    With caching; prefers JSON-only; falls back to "extract JSON from text" on failure.
    """
    cache_key = _payload_cache_key(input_payload)
    p_cache = cfg.outputs_dir / "raw" / f"cache_{cache_key}.json"
    if p_cache.exists():
        try:
            return json.loads(p_cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_INFER + (
                "\n\nIMPORTANT:\n"
                "- Output ONE valid JSON object ONLY; no markdown/explanations.\n"
                "- Expected structure: {'patent_batches':[{'domain','tech_name_raw','patents':[{'patent_id','inferred_codes':[]}]}]}"
            )
        },
        {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)}
    ]

    # First, try JSON-only mode
    try:
        chat = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=cfg.max_tokens
        )
        txt = (chat.choices[0].message.content or "").strip()
        obj = json.loads(txt)
        if isinstance(obj, dict) and "patent_batches" in obj:
            p_cache.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            return obj
    except Exception as e:
        (cfg.outputs_dir/"raw"/f"infer_json_object_err_{int(time.time())}.txt").write_text(repr(e), encoding="utf-8")

    # Fallback: text + extract JSON
    chat = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        messages=messages,
        max_tokens=cfg.max_tokens
    )
    raw = (chat.choices[0].message.content or "")
    (cfg.outputs_dir/"raw"/f"infer_chat_raw_{int(time.time())}.txt").write_text(raw, encoding="utf-8")
    cand = _extract_first_json_object(raw)
    obj = json.loads(cand)
    if isinstance(obj, dict) and "patent_batches" in obj:
        p_cache.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return obj
    return {"patent_batches": []}

# ========= Main Flow =========
def main() -> None:
    parser = argparse.ArgumentParser(description="IPC/CPC Automatic Inference and Local Validation (Resumable, No Duplicates, Auto-batching)")
    parser.add_argument("--mappings-dir", type=str, default="mappings")
    parser.add_argument("--patents-dir",  type=str, default="patents")
    parser.add_argument("--outputs-dir",  type=str, default="outputs")
    parser.add_argument("--techset",      type=str, default="techset_hierarchy.json")

    parser.add_argument("--title-col",    type=str, default="title")
    parser.add_argument("--abstract-col", type=str, default="abstract")
    parser.add_argument("--id-col",       type=str, default="patent id")

    parser.add_argument("--tau-strong",   type=float, default=0.35)
    parser.add_argument("--tau-min",      type=float, default=0.15)
    parser.add_argument("--max-evidence", type=int,   default=5)

    parser.add_argument("--model",        type=str,   default="gpt-4o")
    parser.add_argument("--temperature",  type=float, default=0.2)
    parser.add_argument("--top-p",        type=float, default=1.0)
    parser.add_argument("--max-tokens",   type=int,   default=None)

    parser.add_argument("--batch-size",   type=int,   default=30)
    parser.add_argument("--auto-batch",   action="store_true")
    parser.add_argument("--context-limit",type=int,   default=128000)

    parser.add_argument("--connect-timeout", type=float, default=15)
    parser.add_argument("--read-timeout",    type=float, default=180)
    parser.add_argument("--proxy",           type=str,   default=os.getenv("HTTPS_PROXY") or os.getenv("ALL_PROXY"))
    parser.add_argument("--auto-batch-cap",  type=int,   default=60)

    args = parser.parse_args()

    cfg = Config(
        mappings_dir = Path(args.mappings_dir),
        patents_dir  = Path(args.patents_dir),
        outputs_dir  = Path(args.outputs_dir),
        techset_path = Path(args.techset),

        col_title = args.title_col,
        col_abs   = args.abstract_col,
        col_id    = args.id_col,

        tau_strong = args.tau_strong,
        tau_min    = args.tau_min,
        max_evidence = args.max_evidence,

        model = args.model,
        temperature = args.temperature,
        top_p = args.top_p,
        max_tokens = args.max_tokens,

        batch_size = args.batch_size,
        auto_batch = args.auto_batch,
        context_limit = args.context_limit,

        connect_timeout = args.connect_timeout,
        read_timeout    = args.read_timeout,
        proxy           = args.proxy,
        auto_batch_cap  = args.auto_batch_cap,
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in .env or system environment variables.")

    ensure_outputs(cfg)
    log(f"Mappings: {cfg.mappings_dir}")
    log(f"Patents:  {cfg.patents_dir}")
    log(f"Outputs:  {cfg.outputs_dir}")

    # Read techset
    if not cfg.techset_path.exists():
        raise FileNotFoundError(f"techset_hierarchy.json not found at {cfg.techset_path}")
    techset = json.loads(cfg.techset_path.read_text(encoding="utf-8"))
    # Default domains: "Environmental Perception", "Decision Making and Planning", "Control and Execution", "Support Systems"
    domains = techset.get("domains", ["环境感知","决策规划","控制执行","支撑体系"])
    expected_map = load_expected_code_mapping(cfg, domains)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ckpt = _load_ckpt(cfg)
    all_rows: List[Dict[str, Any]] = []

    for domain in domains:
        domain_dir = cfg.patents_dir / domain
        if not domain_dir.exists():
            log(f"[WARN] Patents dir not found: {domain_dir}")
            continue

        for tech_name_raw, df in read_patent_csvs_for_domain(cfg, domain_dir):
            ck = f"{domain}__{tech_name_raw}"
            if ckpt.get(ck) == "done":
                log(f"[CKPT] skip already done: {ck}")
                continue

            log(f"==> Domain: {domain} | Tech: {tech_name_raw} | Rows: {len(df)}")

            records = df[[cfg.col_id, cfg.col_title, cfg.col_abs]].to_dict("records")
            if cfg.auto_batch:
                bs = choose_dynamic_batch_size(records, cfg)
                log(f"[AUTO-BATCH] choose {bs} (configured {cfg.batch_size})")
            else:
                bs = cfg.batch_size

            batch_infer_collector: List[Dict[str, Any]] = []

            for batch_idx, batch in enumerate(chunk_list(records, bs), start=1):
                payload = {
                    "techset_hierarchy": {
                        "domains": domains,
                        "items": techset.get("items", [])
                    },
                    "patent_batches": [
                        {
                            "domain": domain,
                            "tech_name_raw": tech_name_raw,
                            "patents": [
                                {
                                    "patent_id": str(r.get(cfg.col_id)),
                                    "title":    str(r.get(cfg.col_title, ""))[:2000],
                                    "abstract": str(r.get(cfg.col_abs, ""))[:6000],
                                } for r in batch
                            ]
                        }
                    ]
                }

                try:
                    infer_obj = call_llm_infer(cfg, client, payload)
                except Exception as e_outer:
                    sub_bs = max(10, bs // 2)
                    log(f"[ADAPT] timeout/connection issue on batch {batch_idx}, split to sub-batches of {sub_bs}. Reason: {type(e_outer).__name__}")
                    combined = {"patent_batches": []}
                    pending = list(batch)
                    while pending:
                        sub = pending[:sub_bs]; pending = pending[sub_bs:]
                        sub_payload = {
                            "techset_hierarchy": payload["techset_hierarchy"],
                            "patent_batches": [
                                {
                                    "domain": domain,
                                    "tech_name_raw": tech_name_raw,
                                    "patents": [
                                        {
                                            "patent_id": str(r.get(cfg.col_id)),
                                            "title":    str(r.get(cfg.col_title, ""))[:2000],
                                            "abstract": str(r.get(cfg.col_abs, ""))[:6000],
                                        } for r in sub
                                    ]
                                }
                            ]
                        }
                        try:
                            sub_obj = call_llm_infer(cfg, client, sub_payload)
                            combined["patent_batches"].extend(sub_obj.get("patent_batches", []))
                        except Exception as e_sub:
                            if len(sub) <= 10:
                                log(f"[ADAPT][SKIP] sub-batch failed permanently ({type(e_sub).__name__}). Skipping {len(sub)} items.")
                                continue
                            half = max(5, len(sub)//2)
                            log(f"[ADAPT] sub-batch still failing, bisect into {half} ...")
                            pending = sub[:half] + sub[half:] + pending
                        infer_obj = combined
                    # === End of adaptive call logic ===
            
                # Raw inference file: deterministic naming (avoids duplicates)
                cache_key = _payload_cache_key(payload)[:12]
                raw_path = cfg.outputs_dir/"raw"/f"infer_{_safe_name(domain)}_{_safe_name(tech_name_raw)}_b{batch_idx}_{cache_key}.json"
                if not raw_path.exists():
                    raw_path.write_text(json.dumps(infer_obj, ensure_ascii=False, indent=2), encoding="utf-8")

                if "patent_batches" in infer_obj:
                    batch_infer_collector.extend(infer_obj["patent_batches"])

            # Merge multiple batches
            merged: List[Dict[str, Any]] = []
            for b in batch_infer_collector:
                if b.get("domain")==domain and b.get("tech_name_raw")==tech_name_raw:
                    merged.extend(b.get("patents", []))

            # Local validation
            result = validate_one_tech(cfg, domain, tech_name_raw, merged, expected_map)

            # Summary for a single sub-category (idempotent overwrite)
            one_path = cfg.outputs_dir / "summary" / f"{_safe_name(domain)}__{_safe_name(tech_name_raw)}.csv"
            pd.DataFrame([{
                "domain": domain,
                "tech_name_raw": tech_name_raw,
                "tech_name_expected": result["tech_name_expected"],
                "n_patents": result["stats"]["n_patents"],
                "rate_A": result["stats"]["rate_A"],
                "rate_B": result["stats"]["rate_B"],
                "rate_C": result["stats"]["rate_C"],
                "rate_none": result["stats"]["rate_none"],
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "expected_codes": ";".join(result["expected_codes"])
            }]).to_csv(one_path, index=False, encoding="utf-8")

            all_rows.append({
                "domain": domain,
                "tech_name_raw": tech_name_raw,
                "tech_name_expected": result["tech_name_expected"],
                "n_patents": result["stats"]["n_patents"],
                "rate_A": result["stats"]["rate_A"],
                "rate_B": result["stats"]["rate_B"],
                "rate_C": result["stats"]["rate_C"],
                "rate_none": result["stats"]["rate_none"],
                "verdict": result["verdict"],
                "confidence": result["confidence"]
            })

            # Update checkpoint (for true resumable execution)
            ckpt[ck] = "done"
            _save_ckpt(cfg, ckpt)

    # Global summary (idempotent overwrite)
    pd.DataFrame(all_rows).to_csv(cfg.outputs_dir/"summary_all.csv", index=False, encoding="utf-8")
    log(f"[DONE] {len(all_rows)} tech rows → {cfg.outputs_dir/'summary_all.csv'}")

if __name__ == "__main__":
    main()

