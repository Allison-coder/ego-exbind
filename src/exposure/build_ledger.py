import argparse
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]

EGOCLIP_CSV = None
VERB_MAP_CSV = None
NOUN_MAP_CSV = None

BAD_VERBS = {"be", "have", "do"}

# EK100 noun_class 11 is the hand/body-part class in this project.
# We exclude it because this analysis targets object-centric verb-noun composition.
BODY_PART_NOUN_CLASSES = {11}

BODY_PART_WORDS = {
    "hand", "finger", "fingers", "arm", "arms", "palm", "palms", "thumb", "thumbs",
    "wrist", "wrists", "leg", "legs", "foot", "feet", "knee", "knees",
    "elbow", "elbows", "shoulder", "shoulders"
}

INSTRUMENT_PREPS = {"with", "from", "in", "by", "using", "into", "onto", "on"}


def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("_", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def is_camera_wearer_text(s: str) -> bool:
    return isinstance(s, str) and s.strip().startswith("#C")


def clean_clip_text(s: str):
    s = str(s).strip()

    # Keep only camera-wearer narrations. Drop #O and other non-#C captions.
    if not is_camera_wearer_text(s):
        return None

    s = re.sub(r"^#C\s+C\s+", "", s).strip()
    s = re.sub(r"^#C\s+", "", s).strip()
    return s



# ---------------------------------------------------------------------
# Parser v2 configuration
# ---------------------------------------------------------------------

USE_PRONOUN_BACKOFF = True

# Keep v1 behavior for now, but log noun_anywhere separately.
# If noun_anywhere is noisy, we can later downgrade it to atom-only evidence.
COUNT_ANYWHERE_AS_PAIR = True

PRONOUN_NOUNS = {
    "it", "them", "that", "this", "one", "ones",
    "something", "anything", "everything",
}


def lookup(term, mapping):
    term = norm_text(term)
    candidates = [
        term,
        term.replace("-", " "),
        term.replace(" ", "-"),
    ]
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def load_maps():
    verb_map = dict(pd.read_csv(VERB_MAP_CSV).values)
    noun_map = dict(pd.read_csv(NOUN_MAP_CSV).values)
    verb_map = {str(k): int(v) for k, v in verb_map.items()}
    noun_map = {str(k): int(v) for k, v in noun_map.items()}
    return verb_map, noun_map


def is_body_part_word(tok) -> bool:
    return norm_text(tok.lemma_) in BODY_PART_WORDS or norm_text(tok.text) in BODY_PART_WORDS


def is_instrument_bodypart(tok) -> bool:
    """
    Skip prepositional body parts such as:
    with his hand, from her hand, in hand, by hand.
    These are usually instruments/locations, not the acted-upon object.
    """
    if not is_body_part_word(tok):
        return False

    if tok.dep_ == "pobj":
        prep = norm_text(tok.head.lemma_)
        if prep in INSTRUMENT_PREPS:
            return True

    return False



def map_noun_token_or_phrase(token, doc, noun_map):
    """
    Map a noun token or its containing noun phrase to an EK100 noun class.

    v2 returns:
        noun_class, noun_source

    noun_source:
        noun_phrase
        noun_head
        noun_token_in_chunk
        noun_token
        body_part
        noun_oov
    """

    def clean_phrase(s):
        return " ".join(str(s).lower().replace("-", " ").split())

    def strip_determiners(words):
        dets = {
            "a", "an", "the",
            "his", "her", "my", "your", "their", "our", "its",
            "this", "that", "these", "those"
        }
        return [w for w in words if w not in dets]

    def lookup_noun(term):
        term = clean_phrase(term)
        if not term:
            return None
        n_class = lookup(term, noun_map)
        if n_class is None:
            return None
        if n_class in BODY_PART_NOUN_CLASSES:
            return None
        return n_class

    # Word-level body-part filter.
    if is_body_part_word(token):
        return None, "body_part"

    # 1. Full noun chunk and de-determined noun chunk.
    for chunk in doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            raw_phrase = clean_phrase(chunk.text)
            words = raw_phrase.split()
            no_det_phrase = " ".join(strip_determiners(words))

            # Try without determiners first: "the washing machine" -> "washing machine".
            phrases = [no_det_phrase, raw_phrase]

            # D1 normalization candidates appended last, preserving existing priority.
            for phrase0 in [no_det_phrase, raw_phrase]:
                if not phrase0:
                    continue

                compact0 = phrase0.replace(" ", "")
                if compact0 and compact0 not in phrases:
                    phrases.append(compact0)

                parts0 = phrase0.split()
                if len(parts0) == 2:
                    colon0 = parts0[1] + ":" + parts0[0]
                    if colon0 not in phrases:
                        phrases.append(colon0)

            for phrase in phrases:
                n_class = lookup_noun(phrase)
                if n_class is not None:
                    return n_class, "noun_phrase"

            # 2. Head noun fallback.
            # This approximates EK-style compound reduction:
            # "pizza cutter" -> "cutter", "cabinet door" -> "door".
            head = clean_phrase(chunk.root.lemma_)
            if head and head not in BODY_PART_WORDS:
                n_class = lookup_noun(head)
                if n_class is not None:
                    return n_class, "noun_head"

            # 3. Right-to-left token fallback inside the chunk.
            for tok in reversed(list(chunk)):
                for cand in [clean_phrase(tok.lemma_), clean_phrase(tok.text)]:
                    if not cand or cand in BODY_PART_WORDS:
                        continue
                    n_class = lookup_noun(cand)
                    if n_class is not None:
                        return n_class, "noun_token_in_chunk"

    # 4. Token fallback.
    for cand in [clean_phrase(token.lemma_), clean_phrase(token.text)]:
        if not cand or cand in BODY_PART_WORDS:
            continue
        n_class = lookup_noun(cand)
        if n_class is not None:
            return n_class, "noun_token"

    return None, "noun_oov"


def parse_doc_to_pair(doc, verb_map, noun_map, prev_noun_class=None):
    """
    Parse one EgoClip caption into one EK100-style verb-noun pair.

    v2 returns:
        pair, status, meta

    meta includes:
        parse_path
        noun_source
        verb_lemma
        noun_text
        cache_noun_class
        pronoun_skipped
    """

    meta = {
        "parse_path": "",
        "noun_source": "",
        "verb_lemma": "",
        "noun_text": "",
        "cache_noun_class": None,
        "pronoun_skipped": False,
    }
    saw_unmapped_nonpronoun_dobj = False

    # 1. First valid verb, following v1.
    verb_tok = None
    for tok in doc:
        if tok.pos_ == "VERB":
            lemma = norm_text(tok.lemma_)
            if lemma not in BAD_VERBS:
                verb_tok = tok
                break

    if verb_tok is None:
        return None, "no_verb", meta

    verb = norm_text(verb_tok.lemma_)
    meta["verb_lemma"] = verb

    v_class = lookup(verb, verb_map)
    object_deps = {"dobj", "obj", "attr", "oprd"}

    def is_pronoun(tok):
        lemma = norm_text(tok.lemma_)
        text = norm_text(tok.text)
        return tok.pos_ == "PRON" or lemma in PRONOUN_NOUNS or text in PRONOUN_NOUNS

    def is_backoff_eligible_pronoun(tok):
        if tok.dep_ in {"poss", "det", "nmod:poss"}:
            return False
        return norm_text(tok.text) in {"it", "they", "them", "that", "this", "one"}

    def try_backoff(tok, path):
        if USE_PRONOUN_BACKOFF and prev_noun_class is not None and v_class is not None:
            if isinstance(prev_noun_class, tuple):
                prev_cls, prev_seed = prev_noun_class
            else:
                prev_cls, prev_seed = prev_noun_class, ""
            meta_local = {
                "parse_path": path,
                "noun_source": "previous_narration",
                "verb_lemma": verb,
                "noun_text": norm_text(tok.text),
                "cache_noun_class": None,  # critical: backoff consumes cache but does not refresh it
                "pronoun_skipped": True,
                "backoff_seed_text": str(prev_seed)[:60],
            }
            return (v_class, int(prev_cls)), meta_local
        return None

    def try_noun(tok, path, allow_pair=True):
        if norm_text(tok.lemma_) == verb:
            return None
        if is_instrument_bodypart(tok):
            return None
        if is_body_part_word(tok):
            return None

        n_class, noun_source = map_noun_token_or_phrase(tok, doc, noun_map)
        if n_class is None:
            return None

        # If a pronoun was skipped earlier, mark the fallback path.
        final_path = path
        if meta.get("pronoun_skipped") and path in {"noun_after_verb", "noun_anywhere"}:
            final_path = path + "_after_pronoun_skip"

        meta_local = {
            "parse_path": final_path,
            "noun_source": noun_source,
            "verb_lemma": verb,
            "noun_text": norm_text(tok.text),
            "cache_noun_class": None,  # D4: fallback paths must not refresh previous-narration cache
            "cache_allowed": False,
            "pronoun_skipped": bool(meta.get("pronoun_skipped")),
        }

        if allow_pair and v_class is not None:
            return (v_class, n_class), meta_local

        # Used for verb_oov cache refresh: noun evidence only, no pair.
        return None, meta_local

    def try_syntactic_object(tok, allow_pair=True):
        nonlocal saw_unmapped_nonpronoun_dobj

        if norm_text(tok.lemma_) == verb:
            return None
        if is_instrument_bodypart(tok):
            return None
        if is_body_part_word(tok):
            return None

        n_class, noun_source = map_noun_token_or_phrase(tok, doc, noun_map)
        if n_class is None:
            if tok.pos_ != "PRON":
                saw_unmapped_nonpronoun_dobj = True
            return None

        meta_local = {
            "parse_path": "syntactic_object",
            "noun_source": noun_source,
            "verb_lemma": verb,
            "noun_text": norm_text(tok.text),
            "cache_noun_class": int(n_class),
            "pronoun_skipped": bool(meta.get("pronoun_skipped")),
        }

        if allow_pair and v_class is not None:
            return (v_class, n_class), meta_local

        return None, meta_local

    # 2. Prefer syntactic object attached to selected verb.
    for tok in doc:
        if tok.head.i == verb_tok.i and tok.pos_ in {"NOUN", "PROPN", "PRON"}:
            if tok.dep_ in object_deps:
                if is_pronoun(tok):
                    meta["pronoun_skipped"] = True
                    if not is_backoff_eligible_pronoun(tok):
                        continue
                    result = try_backoff(tok, "pronoun_backoff_object")
                    if result is not None:
                        pair, meta_local = result
                        meta.update(meta_local)
                        return pair, "mapped_backoff", meta
                    continue

                result = try_syntactic_object(tok, allow_pair=(v_class is not None))
                if result is not None:
                    pair, meta_local = result
                    meta.update(meta_local)
                    if v_class is None:
                        return None, "verb_oov_cache_noun", meta
                    return pair, "mapped_dobj", meta

    # 3. Fallback: first mappable noun after the verb.
    for tok in doc:
        if tok.i <= verb_tok.i:
            continue
        if tok.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue

        if is_pronoun(tok):
            meta["pronoun_skipped"] = True
            if not is_backoff_eligible_pronoun(tok):
                continue
            result = try_backoff(tok, "pronoun_backoff_after_verb")
            if result is not None:
                pair, meta_local = result
                meta.update(meta_local)
                return pair, "mapped_backoff", meta
            continue

        result = try_noun(tok, "noun_after_verb", allow_pair=(v_class is not None))
        if result is not None:
            pair, meta_local = result
            meta.update(meta_local)
            if v_class is None:
                return None, "verb_oov_cache_noun", meta
            if saw_unmapped_nonpronoun_dobj:
                meta["marginal_only"] = True
                meta["pair_count_allowed"] = False
                return pair, "mapped_after_marginal_only", meta
            return pair, "mapped_after", meta

    # 4. Last fallback: first mappable noun anywhere.
    for tok in doc:
        if tok.pos_ not in {"NOUN", "PROPN"}:
            continue

        result = try_noun(tok, "noun_anywhere", allow_pair=(v_class is not None))
        if result is not None:
            pair, meta_local = result
            meta.update(meta_local)

            if v_class is None:
                return None, "verb_oov_cache_noun", meta

            if COUNT_ANYWHERE_AS_PAIR:
                if saw_unmapped_nonpronoun_dobj:
                    meta["marginal_only"] = True
                    meta["pair_count_allowed"] = False
                    return pair, "mapped_anywhere_marginal_only", meta
                return pair, "mapped_anywhere", meta
            return None, "noun_anywhere_pair_not_counted", meta

    if v_class is None:
        return None, "verb_oov", meta

    return None, "noun_oov", meta


def save_part(
    part_dir,
    part_idx,
    pair_counter,
    pair_path_counter,
    marginal_only_counter,
    status_counter,
    parse_path_counter,
    noun_source_counter,
    pronoun_skip_counter,
    rows_in_part,
    total_rows,
):
    part_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [
        {"verb_class": v, "noun_class": n, "freq": c}
        for (v, n), c in pair_counter.items()
    ]
    pair_path_rows = [
        {"verb_class": v, "noun_class": n, "parse_path": path, "freq": c}
        for (v, n, path), c in pair_path_counter.items()
    ]
    # NOT pair evidence — marginal aggregation only.
    # Downstream f(v,n) must never read this table as pair frequency.
    marginal_only_rows = [
        {"verb_class": v, "noun_class": n, "parse_path": path, "freq": c}
        for (v, n, path), c in marginal_only_counter.items()
    ]
    status_rows = [
        {"status": k, "count": v}
        for k, v in status_counter.items()
    ]
    parse_path_rows = [
        {"parse_path": k, "count": v}
        for k, v in parse_path_counter.items()
    ]
    noun_source_rows = [
        {"noun_source": k, "count": v}
        for k, v in noun_source_counter.items()
    ]
    pronoun_skip_rows = [
        {"pronoun_skip_type": k, "count": v}
        for k, v in pronoun_skip_counter.items()
    ]

    pd.DataFrame(pair_rows).to_csv(part_dir / f"pairs_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(pair_path_rows).to_csv(part_dir / f"pairs_by_path_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(marginal_only_rows).to_csv(part_dir / f"marginal_only_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(status_rows).to_csv(part_dir / f"status_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(parse_path_rows).to_csv(part_dir / f"parse_path_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(noun_source_rows).to_csv(part_dir / f"noun_source_part_{part_idx:05d}.csv", index=False)
    pd.DataFrame(pronoun_skip_rows).to_csv(part_dir / f"pronoun_skip_part_{part_idx:05d}.csv", index=False)

    meta_fp = part_dir / f"meta_part_{part_idx:05d}.txt"
    with open(meta_fp, "w") as f:
        f.write(f"part_idx={part_idx}\n")
        f.write(f"rows_in_part={rows_in_part}\n")
        f.write(f"total_rows_after_part={total_rows}\n")

    print(f"[SAVE] part={part_idx}, rows_in_part={rows_in_part}, total_rows={total_rows}", flush=True)


def merge_count_parts(part_dir, pattern, key_col):
    files = sorted(part_dir.glob(pattern))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[key_col, "count"])

    out = pd.concat(dfs, ignore_index=True)
    out = (
        out.groupby(key_col, as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    return out


def merge_pair_path_parts(part_dir):
    files = sorted(part_dir.glob("pairs_by_path_part_*.csv"))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["verb_class", "noun_class", "parse_path", "freq"])

    out = pd.concat(dfs, ignore_index=True)
    out = (
        out.groupby(["verb_class", "noun_class", "parse_path"], as_index=False)["freq"]
        .sum()
        .sort_values("freq", ascending=False)
    )
    return out


def merge_marginal_only_parts(part_dir):
    files = sorted(part_dir.glob("marginal_only_part_*.csv"))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["verb_class", "noun_class", "parse_path", "freq"])

    out = pd.concat(dfs, ignore_index=True)
    out = (
        out.groupby(["verb_class", "noun_class", "parse_path"], as_index=False)["freq"]
        .sum()
        .sort_values("freq", ascending=False)
    )
    return out


def merge_parts(part_dir, out_dir, run_name):
    pair_files = sorted(part_dir.glob("pairs_part_*.csv"))

    if not pair_files:
        print("[MERGE] no pair parts found")
        return

    pair_dfs = []
    for fp in pair_files:
        df = pd.read_csv(fp)
        if len(df) > 0:
            pair_dfs.append(df)

    if pair_dfs:
        pair_df = pd.concat(pair_dfs, ignore_index=True)
        pair_df = (
            pair_df.groupby(["verb_class", "noun_class"], as_index=False)["freq"]
            .sum()
            .sort_values("freq", ascending=False)
        )
    else:
        pair_df = pd.DataFrame(columns=["verb_class", "noun_class", "freq"])

    pair_path_df = merge_pair_path_parts(part_dir)
    # NOT pair evidence — marginal aggregation only.
    # Downstream f(v,n) must never read this table as pair frequency.
    marginal_only_df = merge_marginal_only_parts(part_dir)
    status_df = merge_count_parts(part_dir, "status_part_*.csv", "status")
    parse_path_df = merge_count_parts(part_dir, "parse_path_part_*.csv", "parse_path")
    noun_source_df = merge_count_parts(part_dir, "noun_source_part_*.csv", "noun_source")
    pronoun_skip_df = merge_count_parts(part_dir, "pronoun_skip_part_*.csv", "pronoun_skip_type")

    # D3: seen verb/noun marginals include demoted marginal-only evidence,
    # while pair_df remains strict pair evidence only.
    marginal_for_seen_df = marginal_only_df[["verb_class", "noun_class", "freq"]]
    seen_source_df = pd.concat([pair_df, marginal_for_seen_df], ignore_index=True)

    verb_df = (
        seen_source_df.groupby("verb_class", as_index=False)["freq"]
        .sum()
        .sort_values("freq", ascending=False)
    )
    noun_df = (
        seen_source_df.groupby("noun_class", as_index=False)["freq"]
        .sum()
        .sort_values("freq", ascending=False)
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    pair_out = out_dir / f"egoclip_ek100_pair_freq_{run_name}.csv"
    pair_path_out = out_dir / f"egoclip_ek100_pair_freq_by_path_{run_name}.csv"
    marginal_only_out = out_dir / f"egoclip_ek100_marginal_only_{run_name}.csv"
    verb_out = out_dir / f"egoclip_ek100_seen_verbs_{run_name}.csv"
    noun_out = out_dir / f"egoclip_ek100_seen_nouns_{run_name}.csv"
    status_out = out_dir / f"egoclip_ek100_mapping_status_{run_name}.csv"
    parse_path_out = out_dir / f"egoclip_ek100_parse_path_counts_{run_name}.csv"
    noun_source_out = out_dir / f"egoclip_ek100_noun_source_counts_{run_name}.csv"
    pronoun_skip_out = out_dir / f"egoclip_ek100_pronoun_skip_counts_{run_name}.csv"

    pair_df.to_csv(pair_out, index=False)
    pair_path_df.to_csv(pair_path_out, index=False)
    marginal_only_df.to_csv(marginal_only_out, index=False)
    verb_df.to_csv(verb_out, index=False)
    noun_df.to_csv(noun_out, index=False)
    status_df.to_csv(status_out, index=False)
    parse_path_df.to_csv(parse_path_out, index=False)
    noun_source_df.to_csv(noun_source_out, index=False)
    pronoun_skip_df.to_csv(pronoun_skip_out, index=False)

    print("\n[MERGE DONE]")
    print("pair_out:", pair_out)
    print("pair_path_out:", pair_path_out)
    print("marginal_only_out:", marginal_only_out)
    print("verb_out:", verb_out)
    print("noun_out:", noun_out)
    print("status_out:", status_out)
    print("parse_path_out:", parse_path_out)
    print("noun_source_out:", noun_source_out)
    print("pronoun_skip_out:", pronoun_skip_out)

    print("\nStatus:")
    print(status_df.to_string(index=False))

    print("\nParse paths:")
    print(parse_path_df.to_string(index=False))

    print("\nNoun sources:")
    print(noun_source_df.to_string(index=False))

    print("\nPronoun skipped:")
    print(pronoun_skip_df.to_string(index=False))

    print("\nTop 30 pairs:")
    print(pair_df.head(30).to_string(index=False))


def main():
    global EGOCLIP_CSV, VERB_MAP_CSV, NOUN_MAP_CSV

    parser = argparse.ArgumentParser()
    parser.add_argument("--egoclip-csv", type=Path, required=True)
    parser.add_argument("--verb-map-csv", type=Path, required=True)
    parser.add_argument("--noun-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50000, help="0 means full data")
    parser.add_argument("--run-name", type=str, default="v2_smoke50k")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--part-size", type=int, default=500000)
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit

    EGOCLIP_CSV = args.egoclip_csv
    VERB_MAP_CSV = args.verb_map_csv
    NOUN_MAP_CSV = args.noun_map_csv

    out_dir = args.output_dir
    part_dir = out_dir / f"egoclip_map_parts_{args.run_name}"

    print("SCRIPT:", SCRIPT)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("EGOCLIP_CSV:", EGOCLIP_CSV, EGOCLIP_CSV.exists())
    print("VERB_MAP_CSV:", VERB_MAP_CSV, VERB_MAP_CSV.exists())
    print("NOUN_MAP_CSV:", NOUN_MAP_CSV, NOUN_MAP_CSV.exists())
    print("OUT_DIR:", out_dir)
    print("PART_DIR:", part_dir)
    print("limit:", limit)
    print("chunksize:", args.chunksize)
    print("batch_size:", args.batch_size)
    print("part_size:", args.part_size)

    verb_map, noun_map = load_maps()
    print("verb map size:", len(verb_map))
    print("noun map size:", len(noun_map))

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print("spaCy loaded:", spacy.__version__)

    pair_counter = Counter()
    pair_path_counter = Counter()
    marginal_only_counter = Counter()
    status_counter = Counter()
    parse_path_counter = Counter()
    noun_source_counter = Counter()
    pronoun_skip_counter = Counter()

    # Cross-chunk cache for previous-narration noun.
    # Key includes narration_source so pass_1 and pass_2 do not contaminate each other.
    prev_noun_by_stream = {}

    raw_rows_read = 0
    total_rows = 0
    rows_in_part = 0
    part_idx = 0
    t0 = time.time()

    reader = pd.read_csv(
        EGOCLIP_CSV,
        sep="\t",
        error_bad_lines=False,
        usecols=["video_uid", "narration_source", "narration_ind", "narration_time", "clip_text"],
        chunksize=args.chunksize,
    )

    for chunk_idx, chunk in enumerate(reader):
        raw_rows_read += len(chunk)

        if limit is not None:
            remaining = limit - total_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)

        # Sort within chunk. Cache persists across chunks.
        chunk["_narration_time_num"] = pd.to_numeric(chunk["narration_time"], errors="coerce")
        chunk["_narration_ind_num"] = pd.to_numeric(chunk["narration_ind"], errors="coerce")
        chunk = chunk.sort_values(
            ["video_uid", "narration_source", "_narration_time_num", "_narration_ind_num"],
            kind="mergesort"
        )

        records = []
        dropped_other = 0

        for _, row in chunk.iterrows():
            cleaned = clean_clip_text(str(row["clip_text"]))
            if cleaned is None or cleaned == "":
                dropped_other += 1
                continue

            records.append(
                {
                    "video_uid": str(row["video_uid"]),
                    "narration_source": str(row["narration_source"]),
                    "text": cleaned,
                }
            )

        status_counter["other_person_or_empty"] += dropped_other

        docs = nlp.pipe([r["text"] for r in records], batch_size=args.batch_size)

        for rec, doc in zip(records, docs):
            stream_key = (rec["video_uid"], rec["narration_source"])

            try:
                pair, status, meta = parse_doc_to_pair(
                    doc,
                    verb_map,
                    noun_map,
                    prev_noun_class=prev_noun_by_stream.get(stream_key),
                )
            except Exception:
                pair, status, meta = None, "parse_error", {
                    "parse_path": "parse_error",
                    "noun_source": "parse_error",
                    "cache_noun_class": None,
                    "pronoun_skipped": False,
                }

            status_counter[status] += 1

            parse_path = meta.get("parse_path", "")
            noun_source = meta.get("noun_source", "")
            pronoun_skipped = bool(meta.get("pronoun_skipped", False))

            if parse_path:
                parse_path_counter[parse_path] += 1
            if noun_source:
                noun_source_counter[noun_source] += 1
            if pronoun_skipped:
                pronoun_skip_counter[parse_path or status] += 1

            if pair is not None:
                v_class, n_class = pair
                pair_count_allowed = meta.get("pair_count_allowed", True)

                if pair_count_allowed:
                    pair_counter[pair] += 1
                    if parse_path:
                        pair_path_counter[(v_class, n_class, parse_path)] += 1
                else:
                    marginal_only_counter[(v_class, n_class, parse_path)] += 1

            # Critical cache rule:
            # only directly observed nouns refresh the cache.
            # previous_narration backoff consumes cache but does not refresh it.
            cache_noun_class = meta.get("cache_noun_class", None)
            if cache_noun_class is not None and noun_source != "previous_narration":
                seed_text = str(rec.get("text", ""))[:60]
                prev_noun_by_stream[stream_key] = (int(cache_noun_class), seed_text)

            total_rows += 1
            rows_in_part += 1

            if rows_in_part >= args.part_size:
                save_part(
                    part_dir,
                    part_idx,
                    pair_counter,
                    pair_path_counter,
                    marginal_only_counter,
                    status_counter,
                    parse_path_counter,
                    noun_source_counter,
                    pronoun_skip_counter,
                    rows_in_part,
                    total_rows,
                )
                part_idx += 1
                pair_counter = Counter()
                pair_path_counter = Counter()
                marginal_only_counter = Counter()
                status_counter = Counter()
                parse_path_counter = Counter()
                noun_source_counter = Counter()
                pronoun_skip_counter = Counter()
                rows_in_part = 0

        elapsed = time.time() - t0
        print(
            f"[PROGRESS] chunk={chunk_idx+1}, raw_rows_read={raw_rows_read}, "
            f"total_rows={total_rows}, current_part_rows={rows_in_part}, "
            f"dropped_other={dropped_other}, elapsed_min={elapsed/60:.2f}",
            flush=True,
        )

        if limit is not None and total_rows >= limit:
            break

    if rows_in_part > 0 or len(pair_counter) > 0 or len(marginal_only_counter) > 0 or len(status_counter) > 0:
        save_part(
            part_dir,
            part_idx,
            pair_counter,
            pair_path_counter,
            marginal_only_counter,
            status_counter,
            parse_path_counter,
            noun_source_counter,
            pronoun_skip_counter,
            rows_in_part,
            total_rows,
        )

    merge_parts(part_dir, out_dir, args.run_name)


if __name__ == "__main__":
    main()
