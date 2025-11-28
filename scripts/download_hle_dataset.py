#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import gzip

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found.\nInstall: pip install datasets")
    raise SystemExit(1)


KEEP_KEYS = [
    "id", "question", "image", "answer", "answer_type",
    "author_name", "rationale", "raw_subject", "category", "canary",
]

def sanitize_text(s):
    if isinstance(s, str):
        return s.encode("utf-8", "replace").decode("utf-8")
    return s

def to_serializable(item):
    out = {}
    for k in KEEP_KEYS:
        v = item.get(k)
        if k == "image" and not isinstance(v, str):
            v = None
        out[k] = sanitize_text(v) if v is not None else None
    return out

def save_json(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_jsonl(dataset, path, gzip_out=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = (lambda p, m: gzip.open(p, m, encoding="utf-8")) if gzip_out else (lambda p, m: open(p, m, encoding="utf-8"))
    with opener(path, "wt") as f:
        for item in dataset:
            rec = to_serializable(item)
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def download_hle(output_path: str, fmt: str = "jsonl", gzip_out: bool = False):
    print("=" * 70)
    print("Downloading Humanity's Last Exam (HLE) Dataset")
    print("=" * 70)
    print("\n📥 Loading dataset from HuggingFace: cais/hle\n")
    ds = load_dataset("cais/hle", split="test", trust_remote_code=True)
    print(f"✅ Loaded! Total questions: {len(ds)}\n")

    # quick type summary
    if len(ds) > 0:
        first = ds[0]
        print("📊 Sample schema:")
        for k in first.keys():
            print(f"  - {k}: {type(first[k]).__name__}")
        print()

    # counts
    text_only = multimodal = 0
    for item in ds:
        has_img = bool(item.get("images") or item.get("image") or item.get("image_path"))
        multimodal += int(has_img)
        text_only += int(not has_img)
    print("📈 Question types:")
    print(f"  - Text-only: {text_only}")
    print(f"  - Multimodal: {multimodal}\n")

    out_path = Path(output_path)
    if fmt == "json":
        # build a compact list then dump once
        safe_items = [to_serializable(x) for x in ds]
        print(f"💾 Saving JSON to: {out_path}")
        save_json(safe_items, out_path)
    else:
        # stream-friendly JSONL
        print(f"💾 Saving JSONL to: {out_path} (gzip={gzip_out})")
        save_jsonl(ds, out_path, gzip_out=gzip_out)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved. File size: {size_mb:.2f} MB")
    print("\nNext steps:")
    print("  1) Use head/tail/grep/jq on the .jsonl for quick inspection")
    print("  2) Filter text-only for inference, e.g., in your prepare script\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download HLE and save as JSON/JSONL")
    parser.add_argument(
        "--output",
        type=str,
        default="inference/eval_data/hle_full.jsonl",
        help="Output path (use .jsonl or .json). Default: inference/eval_data/hle_full.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default=None,
        help="Force output format. If omitted, inferred from file extension.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Gzip output (only applies to JSONL). Produces *.jsonl.gz",
    )
    args = parser.parse_args()

    # infer format from extension if not provided
    out = Path(args.output)
    fmt = args.format
    if fmt is None:
        ext = out.suffix.lower()
        if ext == ".json":
            fmt = "json"
        elif ext == ".gz" and out.name.endswith(".jsonl.gz"):
            fmt = "jsonl"
        else:
            fmt = "jsonl"  # default

    # if gzip requested and user didn't add .gz, append it
    if args.gzip and fmt == "jsonl" and not str(out).endswith(".gz"):
        out = out.with_suffix(out.suffix + ".gz")  # e.g. .jsonl.gz

    ok = download_hle(str(out), fmt=fmt, gzip_out=args.gzip)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
