import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.binding.adapter import DualResidualAdapters


def norm_text(x):
    return " ".join(str(x).strip().lower().split())


def l2(x, eps=1e-8):
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def load_model(path, device):
    model = DualResidualAdapters().to(device).eval()
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", path)
    print("missing:", missing)
    print("unexpected:", unexpected)
    return model


def bootstrap_corr(x, y, n_boot=10000, seed=2026):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(n=n, pearson=np.nan, spearman=np.nan, sp_ci_low=np.nan, sp_ci_high=np.nan, status="bad")
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            continue
        boots.append(spearmanr(bx, by)[0])
    boots = np.asarray(boots)
    return dict(
        n=n,
        pearson=float(pr),
        spearman=float(sr),
        sp_ci_low=float(np.percentile(boots, 2.5)),
        sp_ci_high=float(np.percentile(boots, 97.5)),
        status="ok",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large-probe", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--text-embeds", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    probe = pd.read_csv(args.large_probe)
    cache = torch.load(args.eval_cache, map_location="cpu")
    val = pd.read_csv(Path(args.meta_dir) / "EPIC_100_retrieval_test.csv")
    text_obj = torch.load(args.text_embeds, map_location="cpu")

    vid_embeds = cache["vid_embeds"].float()
    arr = cache["arr_embeds"].cpu().numpy().astype(int)

    nid_to_cache_row = {}
    for cache_i, val_i in enumerate(arr):
        nid = str(val.iloc[int(val_i)]["narration_id"])
        nid_to_cache_row[nid] = int(cache_i)

    missing_vid = [x for x in probe["narration_id"].astype(str) if x not in nid_to_cache_row]
    if missing_vid:
        raise RuntimeError(f"probe narration_id not in eval cache: n={len(missing_vid)} first={missing_vid[:10]}")

    video_rows = [nid_to_cache_row[x] for x in probe["narration_id"].astype(str)]
    video_pre = vid_embeds[video_rows].float()

    text_embeds = text_obj["text_embeds"].float()
    text_map = {}
    for i, txt in enumerate(text_obj["texts"]):
        text_map.setdefault(norm_text(txt), i)

    def resolve_text_col(col):
        missing = []
        idx = []
        for x in probe[col].astype(str):
            k = norm_text(x)
            if k not in text_map:
                missing.append(x)
                idx.append(-1)
            else:
                idx.append(text_map[k])
        if missing:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            miss_path = out / f"missing_{col}_external_texts.csv"
            pd.DataFrame({col: missing}).drop_duplicates().to_csv(miss_path, index=False)
            raise RuntimeError(f"{col} missing in external text embeddings: n={len(missing)} first={missing[:10]} wrote={miss_path}")
        return idx

    pos_idx = resolve_text_col("positive_text")
    noun_idx = resolve_text_col("noun_negative_text")
    verb_idx = resolve_text_col("verb_negative_text")

    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        v0 = video_pre.to(device)
        t0 = text_embeds.to(device)

        v = l2(model.video_adapter(v0))
        t = l2(model.text_adapter(t0))

        s_pos = (v * t[pos_idx]).sum(dim=1)
        s_noun = (v * t[noun_idx]).sum(dim=1)
        s_verb = (v * t[verb_idx]).sum(dim=1)

        noun_margin = s_pos - s_noun
        verb_margin = s_pos - s_verb

    per = probe.copy()
    per["cache_row"] = video_rows
    per["score_positive"] = s_pos.cpu().numpy()
    per["score_noun_negative"] = s_noun.cpu().numpy()
    per["score_verb_negative"] = s_verb.cpu().numpy()
    per["noun_margin"] = noun_margin.cpu().numpy()
    per["verb_margin"] = verb_margin.cpu().numpy()
    per["noun_correct"] = (noun_margin > 0).cpu().numpy()
    per["verb_correct"] = (verb_margin > 0).cpu().numpy()

    if "log10_f_pair" not in per.columns:
        per["log10_f_pair"] = np.log10(per["f_positive_pair"].astype(float))

    rows = []
    for metric, mcol in [("noun_margin", "noun_margin"), ("verb_margin", "verb_margin")]:
        r = bootstrap_corr(per[mcol], per["log10_f_pair"])
        r.update({"metric": metric, "xcol": "log10_f_pair"})
        rows.append(r)
    rho = pd.DataFrame(rows)

    summary = {
        "n": int(len(per)),
        "video_col": "narration_id->arr_embeds",
        "text_col": "external encoded probe template texts",
        "noun_acc": float(per["noun_correct"].mean()),
        "verb_acc": float(per["verb_correct"].mean()),
        "noun_margin_mean": float(per["noun_margin"].mean()),
        "verb_margin_mean": float(per["verb_margin"].mean()),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "large_binding_per_probe_v1.csv", index=False)
    rho.to_csv(out / "large_c3_rho_bootstrap_v1.csv", index=False)
    with open(out / "large_binding_summary_v1.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("probe shape:", probe.shape)
    print("video_route: narration_id->arr_embeds")
    print("text_route: external encoded probe template texts")
    print(json.dumps(summary, indent=2))
    print("===== LARGE C3 RHO =====")
    print(rho.to_string(index=False))
    print("LARGE_BINDING_EVAL_PASS")
    print("wrote:", out)


if __name__ == "__main__":
    main()
