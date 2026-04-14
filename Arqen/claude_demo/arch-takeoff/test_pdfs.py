"""
Test script — sends 10 PDF plans from the data folder to the Claude vision API
and prints the raw JSON takeoff results for each.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python test_pdfs.py
"""

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL = "claude-sonnet-4-6-20250131"
MAX_TOKENS = 4096
NUM_PLANS = 10

PROMPT = """\
You are an expert architectural drawing analyst and quantity surveyor.
Simply measure each exterior wall of this architectural plan document.

SCALE INSTRUCTIONS: Use the inputted scale to measure the walls.

MEASUREMENT UNITS: Return all measurements in feet (ft).

DETECTION SCOPE: Identify all exterior walls, and their corresponding facing direction (North, South, East, West). 
Additionally, identify any windows along those walls.

CRITICAL: Return ONLY a valid JSON object — no markdown, no explanation, no code
blocks, nothing else. Pure JSON only.

ACCURACY: Return measurements with high accuracy, and consistency. 
Be sure to double check your work, aligning measurements with the plan and inputted scale

Return exactly this structure:
{
  "detected_scale": "e.g. 1:100 or 1/8 inch = 1 foot",
  "total_area": "e.g. 1534 ft²",
  "units": "imperial",
  "walls": [
    {
      "windows": 0,
      "id": "w1",
      "name": "North Wall",
      "facing": "North",
      "length": "42.0 ft",
      "notes": ""
    }
  ],
}
"""


def load_pdfs(directory: Path, limit: int) -> list[Path]:
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {directory}")
        sys.exit(1)
    selected = pdfs[:limit]
    print(f"Found {len(pdfs)} PDFs, selecting first {len(selected)}:\n")
    for i, p in enumerate(selected, 1):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {i:>2}. {p.name}  ({size_mb:.1f} MB)")
    print()
    return selected


def analyze_pdf(client: anthropic.Anthropic, pdf_path: Path) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {x
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")

    json_match = None
    m = re.search(r"\{[\s\S]*\}", raw_text)
    if m:
        json_match = m.group(0)

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    try:
        parsed = json.loads(json_match) if json_match else {"_raw": raw_text}
    except json.JSONDecodeError:
        parsed = {"_raw": raw_text, "_parse_error": True}

    return {"result": parsed, "usage": usage, "model": response.model}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        print("  e.g.  set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    pdfs = load_pdfs(DATA_DIR, NUM_PLANS)

    all_results = []
    total_input = 0
    total_output = 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Analyzing: {pdf_path.name} ...", flush=True)
        t0 = time.time()
        try:
            result = analyze_pdf(client, pdf_path)
            elapsed = time.time() - t0
            total_input += result["usage"]["input_tokens"]
            total_output += result["usage"]["output_tokens"]

            entry = {
                "file": pdf_path.name,
                "elapsed_sec": round(elapsed, 1),
                **result,
            }
            all_results.append(entry)

            print(f"    Done in {elapsed:.1f}s  "
                  f"(in: {result['usage']['input_tokens']:,} / out: {result['usage']['output_tokens']:,} tokens)")
            print(f"    Scale: {result['result'].get('detected_scale', '?')}  "
                  f"Confidence: {result['result'].get('scale_confidence', '?')}  "
                  f"Rooms: {len(result['result'].get('rooms', []))}")
            print()
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ERROR after {elapsed:.1f}s: {e}\n")
            all_results.append({"file": pdf_path.name, "error": str(e)})

    print("=" * 70)
    print(f"Total tokens — input: {total_input:,}  output: {total_output:,}")
    print("=" * 70)

    out_path = Path(__file__).resolve().parent / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull JSON results saved to: {out_path}")

    print("\n--- JSON OUTPUT (pretty) ---\n")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
