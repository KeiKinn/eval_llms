"""


evaluate.py – **Single-file LLM evaluation tool**
==============================================
The script is self-contained: it installs any missing Python packages
on first run and skips metrics that cannot be loaded.

Metrics implemented
-------------------
| Name                    | Range        | What it measures (↑ = better)                              |
|-------------------------|--------------|------------------------------------------------------------|
| **BLEU-1 / BLEU-4**     | 0 – 1        | Word-level and up-to-4-gram precision                      |
| **METEOR**              | 0 – 1        | Stem/synonym overlap                                       |
| **ROUGE-L**             | 0 – 1        | Longest common subsequence (recall-oriented)               |
| **CIDEr**               | 0 – 10       | TF-IDF-weighted n-gram consensus (caption benchmark)       |
| **CosineSim (Nomic)**   | -1 – 1       | Semantic similarity via *nomic-ai/nomic-embed-text-v1*     |
| **BLEURT-20**           | ~0 – 1       | Neural adequacy + fluency (trained metric)                 |

Deeper info:
* **CIDEr** – consensus n-gram similarity, TF-IDF-weighted (0-10, higher⇑).
  Measures how much a candidate caption uses the same *informative* n-grams
  as the references, giving more weight to rare words.
* **CosineSim (Nomic)** – average cosine similarity between reference and
  prediction embeddings from *nomic-ai/nomic-embed-text-v1* (-1→1, higher⇑).
  Captures overall semantic alignment.
* **BLEURT-20** – a multilingual BERT-based regressor fine-tuned on hundreds-of-thousands of human adequacy ratings plus synthetic noise.  
  Scores usually land between **0 and 1** (higher ⇑ is better).  A BERT/RemBERT model fine-tuned on human judgement data;
  Unlike lexical metrics, BLEURT rewards any well-formed paraphrase that preserves the reference meaning, even if it re-phrases almost every word.
- Download model: wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
- Unzip it: unzip BLUERT-20 

Quick start
-----------
bash
python llm_eval.py                  # uses defaults: dummy_eval_data.csv, etc.
(barplot_comparison.png) in the same folder.


The script writes llm_evaluation_results.csv plus a bar plot

Dependencies auto-installed on first run:
* pycocoevalcap  (for CIDEr)
* sentence-transformers  (for Nomic embeddings)
* NLTK punkt/wordnet data (quiet download)

If a metric cannot be computed you’ll see ERR: … in that cell instead
of a number.


Usage
-----
```bash
python evaluate.py \
       --csv dummy_eval_data.csv \
       --reference answer \
       --preds llm1_prediction llm2_prediction
```
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports & helpers – safe package bootstrap even when the file is named
# `evaluate.py` (avoids self-import shadowing the real library).
# ---------------------------------------------------------------------------

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import BERTScorer          # pip install bert-score

import nltk  # NLTK downloads handled lazily further below

# ---------------------------------------------------------------------------
# ---- 1.  *Critical* bit: pull in the **real** `evaluate` package ----------
# ---------------------------------------------------------------------------

load = None  # will hold evaluate.load if import is successful

if "evaluate" in sys.modules:
    # If our own module inserted itself under that name, remove it temporarily
    if sys.modules["evaluate"].__file__ == __file__:
        sys.modules.pop("evaluate")

# Temporarily move the script dir to the *end* of sys.path so site-packages
# takes precedence for the import we’re about to do.
_cwd = Path(__file__).resolve().parent
if str(_cwd) in sys.path:
    sys.path.remove(str(_cwd))
    _put_back_path = True
else:
    _put_back_path = False

try:
    _eval_mod = importlib.import_module("evaluate")
    load = getattr(_eval_mod, "load", None)
except ModuleNotFoundError:
    # Library not installed – we’ll fall back to classic metrics only.
    load = None
finally:
    if _put_back_path:
        sys.path.insert(0, str(_cwd))  # restore original import order

# ---------------------------------------------------------------------------
# ---- 2. Remaining helpers (package installs, metric wrappers, etc.) ------
# ---------------------------------------------------------------------------

def _pip_install(pkg: str):
    """Install *pkg* quietly via pip (in-process)."""
    subprocess.check_call([
        "python",
        "-m",
        "pip",
        "install",
        pkg,
        "--quiet",
    ])


def _ensure_nltk() -> None:
    for pkg in ("tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg.split("/")[-1], quiet=True)


def _ensure_pycoco() -> None:
    if importlib.util.find_spec("pycocoevalcap") is None:
        print("[i] Installing pycocoevalcap … (for CIDEr/SPICE)")
        _pip_install("pycocoevalcap")
        print("[✓] pycocoevalcap installed")


def _ensure_sentence_transformers() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        print("[i] Installing sentence-transformers … (for embeddings)")
        _pip_install("sentence-transformers")
        print("[✓] sentence-transformers installed")

# ---------------------------------------------------------------------------
# ---- 3. Metric loaders & implementations ---------------------------------
# ---------------------------------------------------------------------------

def _safe_load(metric_name: str):
    """Load a metric from *evaluate*; return ``None`` if unavailable."""
    if load is None:
        return None
    try:
        return load(metric_name)
    except (FileNotFoundError, ModuleNotFoundError):
        print(f"[!] Metric '{metric_name}' not found – skipping.")
        return None


# Dedicated implementations for CIDEr, SPICE and CosineSim  ----------------

def _compute_cider(refs: List[str], preds: List[str]) -> float:
    _ensure_pycoco()
    from pycocoevalcap.cider.cider import Cider  # type: ignore

    gts = {i: [refs[i]] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def _compute_spice(refs: List[str], preds: List[str]) -> float:
    _ensure_pycoco()
    from pycocoevalcap.spice.spice import Spice  # type: ignore

    gts = {i: [refs[i]] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}
    score, _ = Spice().compute_score(gts, res)
    return float(score)


def _compute_cosine_nomic(refs: List[str], preds: List[str]) -> float:
    _ensure_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    ref_inp = [f"search_document: {t}" for t in refs]
    pred_inp = [f"search_query: {t}" for t in preds]

    ref_emb = model.encode(ref_inp, normalize_embeddings=True, batch_size=64)
    pred_emb = model.encode(pred_inp, normalize_embeddings=True, batch_size=64)

    sims = (ref_emb * pred_emb).sum(axis=1)
    return float(np.mean(sims))

# ---------------------------------------------------------------------------
# ---- 4. Core evaluation routine ------------------------------------------
# ---------------------------------------------------------------------------


def evaluate(
    csv_path: os.PathLike | str,
    reference_col: str,
    pred_cols: List[str],
    *,
    save_dir: os.PathLike | str = "./eval_out",
    save_figs: bool = True,
    with_bleurt: bool = False,                # NEW
    bleurt_ckpt: os.PathLike | str = "./BLEURT-20",  # NEW
):
    """Run automatic evaluation on *csv_path* (single reference vs ≥1 models)."""

    # _ensure_nltk()
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if reference_col not in df.columns:
        raise ValueError(f"Reference column '{reference_col}' not found.")

    refs = df[reference_col].astype(str).tolist()

    pred_cols = [c for c in pred_cols if c in df.columns]
    if not pred_cols:
        raise ValueError("None of the specified prediction columns exist in the CSV.")

    # --- BLEURT support --------------------------------------------------------
    def _ensure_bleurt() -> None:
        """Install the `bleurt` wheel if it is missing."""
        if importlib.util.find_spec("bleurt") is None:
            print("[i] Installing bleurt …")
            _pip_install("bleurt==0.0.2")
            print("[✓] bleurt installed")

    def _ensure_bleurt_ckpt(ckpt_path: Path) -> Path:
        """
        Make sure *ckpt_path* exists.
        If the directory/file is missing, pull the official BLEURT-20 checkpoint.
        """
        if ckpt_path.exists():
            return ckpt_path

        ckpt_url = (
            "https://storage.googleapis.com/bleurt-20/BLEURT-20.tgz"
        )  # 1.1 MB tarball
        print(f"[i] Downloading BLEURT-20 checkpoint → {ckpt_path} …")
        tmp_tgz = ckpt_path.with_suffix(".tgz")
        subprocess.check_call(["curl", "-L", "-o", tmp_tgz, ckpt_url])
        subprocess.check_call(["tar", "xzf", tmp_tgz, "-C", ckpt_path.parent])
        tmp_tgz.unlink(missing_ok=True)
        print("[✓] BLEURT-20 ready")
        return ckpt_path
    # ---- metric definitions ---------------------------------------------
    bleu = _safe_load("bleu")
    meteor = _safe_load("meteor")
    rouge = _safe_load("rouge")

    metric_defs: list[tuple[str, Callable[[List[str]], float], float, str]] = []

    if bleu is not None:
        metric_defs += [
            ("BLEU-1", lambda p, _b=bleu: _b.compute(predictions=p, references=[[r] for r in refs], max_order=1)["bleu"], 1.0, "Word precision"),
            ("BLEU-4", lambda p, _b=bleu: _b.compute(predictions=p, references=[[r] for r in refs], max_order=4)["bleu"], 1.0, "Up-to-4-gram"),
        ]
    if meteor is not None:
        metric_defs.append(("METEOR", lambda p, _m=meteor: _m.compute(predictions=p, references=[[r] for r in refs])["meteor"], 1.0, "Stem/synonym recall"))
    if rouge is not None:
        metric_defs.append(("ROUGE-L", lambda p, _r=rouge: _r.compute(predictions=p, references=[[r] for r in refs])["rougeL"], 1.0, "Longest common subsequence"))

    metric_defs += [
        ("CIDEr", lambda p: _compute_cider(refs, p), 10.0, "TF-IDF consensus"),
        # ("SPICE", lambda p: _compute_spice(refs, p), 1.0, "Scene-graph semantics"),
        ("CosineSim(Nomic)", lambda p: _compute_cosine_nomic(refs, p), 1.0, "Embedding similarity"),
    ]
        # -------- optional BLEURT-20 -----------------------------------------
    if with_bleurt:
        _ensure_bleurt()
        from bleurt import score   # import only if really needed

        ckpt = _ensure_bleurt_ckpt(Path(bleurt_ckpt))
        _scorer = score.BleurtScorer(str(ckpt))  # load once, reuse!

        metric_defs.append(
            ("BLEURT-20",
             lambda p, _s=_scorer: float(np.mean(_s.score(references=refs,
                                                          candidates=p))),
             1.0, "Neural semantic similarity")
        )
    # ---- BERTScore -----------------------------------------------------------
    try:
        from bert_score import BERTScorer
    except ImportError:
        _pip_install("bert-score")
        from bert_score import BERTScorer

    def _get_bertscorer():
        """Instantiate once, reuse across models to save GPU / CPU time."""
        # Auto-detect: English → 'bert-base-uncased', else multilingual
        bert_lang = "en" if all(t.isascii() for t in refs[:50]) else "multilingual"
        model_type = "microsoft/deberta-large-mnli" if bert_lang == "en" else "microsoft/mdeberta-v3-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return BERTScorer(
            lang=bert_lang,
            model_type=model_type,
            rescale_with_baseline=True,   # enables IDF re-scaling for fairer scores
            device=device,
        )

    _bertscorer = _get_bertscorer()
    metric_defs.append((
        "BERTScore",
        lambda p, _bs=_bertscorer: float(_bs.score(p, refs)[2].mean().item()),
        1.0,
        "Contextual embedding",
    ))


    # ---- compute scores --------------------------------------------------
    rows = []
    for mname, fn, rng_max, expl in metric_defs:
        row: dict[str, float | str] = {"Metric": mname, "Range": f"0–{rng_max}", "Interpretation": expl}
        for col in pred_cols:
            try:
                row[col] = fn(df[col].astype(str).tolist())
            except Exception as e:  # pylint: disable=broad-except
                row[col] = f"ERR: {e}".split("\n")[0]
        rows.append(row)

    res_df = pd.DataFrame(rows)
    out_csv = save_dir / "llm_evaluation_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("\nEvaluation results (higher is better):\n")
    print(res_df.to_string(index=False))
    print("\n[✓] Saved to", out_csv)

    # ---- plots -----------------------------------------------------------
    if save_figs:
        _plot_bar(res_df, pred_cols, save_dir)
        if len(pred_cols) >= 2:
            _plot_radar(res_df, pred_cols[:2], save_dir)


# ---------------------------------------------------------------------------
# ---- 5. Plot utilities  (replace the two functions only) ------------------
# ---------------------------------------------------------------------------

# ❶ Classify the metrics once so we can style them differently ---------------
_LEXICAL = {
    "BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr",  # add SPICE if you turn it on
}
_EMBED   = {
    "CosineSim(Nomic)", "BLEURT-20", "BERTScore",
}

dark_blue_palette = [
    "#AEC6CF", "#FFB347", "#77DD77", "#CBAACB", "#FFD1DC",
    "#FDFD96", "#B0E0E6", "#D6AEDD", "#FFDAC1", "#E0BBE4",
]

# ------------------------------------------------------------------ BAR PLOT
def _plot_bar(df: pd.DataFrame, model_cols: List[str], save_dir: Path):
    """
    Bar-plot with **stars** (hatch) for lexical metrics and plain bars for
    embedding / neural metrics.
    """
    melted = df.melt(
        id_vars=["Metric"],
        value_vars=model_cols,
        var_name="Model",
        value_name="Score",
    )
    melted["Score"] = pd.to_numeric(melted["Score"], errors="coerce")

    # draw the bars ----------------------------------------------------------
    g = sns.catplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        kind="bar",
        palette=dark_blue_palette[: len(model_cols)],
        height=6,
        aspect=1.6,
        dodge=True,
        errwidth=0,
    )

    # Add hatches (= little stripes) on the lexical bars only ----------------
    ax = g.ax
    # hatch = "*"
    for i, this_bar in enumerate(ax.patches):
        metric = melted.iloc[i // len(model_cols)]["Metric"]
        # if metric in _LEXICAL:
        #     this_bar.set_hatch(hatch)

    ymax = melted["Score"].max(skipna=True)
    ax.set_ylim(0, 1.1 * ymax if ymax and ymax > 1 else 1.1)
    ax.set_title("LLM Evaluation Metrics")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    g.tight_layout()
    outfile = save_dir / "barplot_comparison.png"
    g.savefig(outfile)
    plt.close(g.figure)
    print("[✓] Bar plot saved to", outfile)


# ------------------------------------------------------------------ RADAR PLOT
def _plot_radar(df: pd.DataFrame, two_models: List[str], save_dir: Path):
    """
    Radar plot: lexical points marked with ★, embedding points with ●.
    """
    labels = df["Metric"].tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    colours = dark_blue_palette[: len(two_models)]
    markers_for_metric = ["*" if m in _LEXICAL else "o" for m in labels]

    for col, colour in zip(two_models, colours):
        vals = df[col].astype(float).tolist() + [df[col].iloc[0]]
        ax.plot(angles, vals, color=colour, linewidth=2, label=col)
        # scatter points with per-metric marker style ------------------------
        for ang, v, m in zip(angles, vals, markers_for_metric + [markers_for_metric[0]]):
            ax.scatter([ang], [v], marker=m, color=colour, s=80, zorder=3)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, np.nanmax(df[two_models].to_numpy(dtype=float)) * 1.1)
    ax.set_title("Radar Plot (first two models)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    outfile = save_dir / "radar_plot_comparison.png"
    plt.savefig(outfile)
    plt.close()
    print("[✓] Radar plot saved to", outfile)


# ---------------------------------------------------------------------------
# ---- 6. CLI --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Single-file LLM evaluation tool")
    p.add_argument("--csv", default="dummy_eval_data.csv", help="Input CSV file")
    p.add_argument("--reference", default="answer", help="Reference column name")
    p.add_argument("--preds", nargs="+", default=["llm1_prediction", "llm2_prediction"], help="Prediction column(s)")
    p.add_argument("--no-figs", action="store_true", help="Skip plot generation")
    p.add_argument("--save-dir", default="./eval_out", help="Output directory for results & plots")
    p.add_argument("--with-bleurt", action="store_true",
                   help="Compute BLEURT-20 as well (installs bleurt if needed)")
    p.add_argument("--bleurt-checkpoint", default="./BLEURT-20",
                   help="Folder or model file for BLEURT-20")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        csv_path=args.csv,
        reference_col=args.reference,
        pred_cols=args.preds,
        save_dir=args.save_dir,
        save_figs=not args.no_figs,
        with_bleurt=args.with_bleurt,          # <-- NEW
        bleurt_ckpt=args.bleurt_checkpoint,    # <-- NEW
    )
