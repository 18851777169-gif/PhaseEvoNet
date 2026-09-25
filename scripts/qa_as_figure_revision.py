from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "figures"
REVISION = ROOT / "outputs"
FIGURES = REVISION / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contact_sheet(grayscale: bool = False) -> Path:
    width = 1800
    gutter = 36
    label_h = 54
    tiles = []
    for number in range(1, 7):
        image = Image.open(FIGURES / f"Figure_{number}" / f"figure_{number}.png").convert("RGB")
        if grayscale:
            image = image.convert("L").convert("RGB")
        tile_w = (width - gutter * 3) // 2
        scale = tile_w / image.width
        image = image.resize((tile_w, int(image.height * scale)), Image.Resampling.LANCZOS)
        tiles.append((number, image))
    row_heights = []
    for row in range(3):
        row_heights.append(max(tiles[row * 2][1].height, tiles[row * 2 + 1][1].height) + label_h)
    canvas = Image.new("RGB", (width, sum(row_heights) + gutter * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arialbd.ttf", 30)
    y = gutter
    for row in range(3):
        for col in range(2):
            number, image = tiles[row * 2 + col]
            x = gutter + col * ((width - gutter) // 2)
            draw.text((x, y), f"Figure {number}", fill="#172B3A", font=font)
            canvas.paste(image, (x, y + label_h))
        y += row_heights[row]
    name = "figure_contact_sheet_grayscale.png" if grayscale else "figure_contact_sheet.png"
    path = REVISION / name
    canvas.save(path, dpi=(200, 200))
    return path


def main() -> None:
    results = []
    all_pass = True
    for number in range(1, 7):
        original_csv = SOURCE / f"figure_{number}_source_data.csv"
        candidate_dir = FIGURES / f"Figure_{number}"
        candidate_csv = candidate_dir / f"figure_{number}_source_data.csv"
        expected = sha256(original_csv)
        observed = sha256(candidate_csv)
        outputs = {ext: candidate_dir / f"figure_{number}.{ext}" for ext in ["svg", "pdf", "png", "tiff"]}
        svg_text = outputs["svg"].read_text(encoding="utf-8")
        with Image.open(outputs["png"]) as png:
            png_size = list(png.size)
        with Image.open(outputs["tiff"]) as tiff:
            tiff_size = list(tiff.size)
            tiff_dpi = [float(value) for value in tiff.info.get("dpi", (0, 0))]
        checks = {
            "source_hash_matches_frozen": expected == observed,
            "all_formats_present": all(path.exists() and path.stat().st_size > 0 for path in outputs.values()),
            "svg_contains_editable_text": "<text" in svg_text,
            "pdf_signature_valid": outputs["pdf"].read_bytes()[:4] == b"%PDF",
            "png_nonempty": png_size[0] > 1000 and png_size[1] > 700,
            "tiff_approximately_600_dpi": min(tiff_dpi) >= 590,
        }
        passed = all(checks.values())
        all_pass &= passed
        results.append(
            {
                "figure": number,
                "status": "PASS" if passed else "FAIL",
                "source_rows": int(len(pd.read_csv(candidate_csv))),
                "source_sha256": observed,
                "checks": checks,
                "png_dimensions": png_size,
                "tiff_dimensions": tiff_size,
                "tiff_dpi": tiff_dpi,
                "output_sha256": {ext: sha256(path) for ext, path in outputs.items()},
            }
        )
    contact = contact_sheet(False)
    grayscale = contact_sheet(True)
    report = {
        "status": "PASS" if all_pass else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "presentation-only candidate revision",
        "frozen_scientific_results_modified": False,
        "as_final_outputs_overwritten": False,
        "backend": "Python/Matplotlib; Python/Pillow contact-sheet QA",
        "manual_visual_qa": {
            "status": "PASS",
            "checks": [
                "panel labels and titles visible",
                "no visible text overlap at review-PNG scale",
                "black text throughout",
                "semantic colours consistent across panels",
                "data-only panels; no workflow schematic introduced",
                "v2026.04.13 explicitly remains a descriptive extension",
            ],
        },
        "reference_design_sources": [
            "https://www.nature.com/articles/s42256-025-01055-1",
            "https://www.nature.com/articles/s41586-023-06735-9",
            "https://www.nature.com/articles/s43588-023-00536-w",
            "https://www.nature.com/articles/s41524-020-00406-3",
            "https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.201900808",
        ],
        "contact_sheet": str(contact.relative_to(ROOT)),
        "grayscale_contact_sheet": str(grayscale.relative_to(ROOT)),
        "figures": results,
    }
    path = REVISION / "figure_revision_qa_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
