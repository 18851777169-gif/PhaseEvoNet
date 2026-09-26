"""Automated figure integrity checks; visual review is a separate step."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from phase_evonet.figure_sources import load_manifest, sha256, verify_table


ROOT = Path(__file__).resolve().parents[1]


def contact_sheet_font(size: int = 30) -> ImageFont.FreeTypeFont:
    """Use a font shipped with a declared dependency, not a system-only filename."""
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(path), size)


def contact_sheet(root: Path, grayscale: bool = False) -> Path:
    revision = root / "outputs"
    figures = revision / "figures"
    width, gutter, label_h = 1800, 36, 54
    tiles = []
    for number in range(1, 7):
        with Image.open(figures / f"Figure_{number}/figure_{number}.png") as original:
            image = original.convert("RGB")
        if grayscale:
            image = image.convert("L").convert("RGB")
        tile_w = (width - gutter * 3) // 2
        image = image.resize((tile_w, int(image.height * tile_w / image.width)), Image.Resampling.LANCZOS)
        tiles.append((number, image))
    row_heights = [max(tiles[row * 2][1].height, tiles[row * 2 + 1][1].height) + label_h for row in range(3)]
    canvas = Image.new("RGB", (width, sum(row_heights) + gutter * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = contact_sheet_font()
    y = gutter
    for row in range(3):
        for col in range(2):
            number, tile = tiles[row * 2 + col]
            x = gutter + col * ((width - gutter) // 2)
            draw.text((x, y), f"Figure {number}", fill="#111111", font=font)
            canvas.paste(tile, (x, y + label_h))
        y += row_heights[row] + gutter
    name = "figure_contact_sheet_grayscale.png" if grayscale else "figure_contact_sheet.png"
    path = revision / name
    canvas.save(path, dpi=(200, 200))
    return path


def audit_figures(root: Path) -> dict[str, object]:
    results = []
    errors = []
    try:
        registered = load_manifest(root)
    except (OSError, ValueError) as exc:
        registered = {}
        errors.append(str(exc))
    for number, expected in registered.items():
        candidate_dir = root / f"outputs/figures/Figure_{number}"
        outputs = {ext: candidate_dir / f"figure_{number}.{ext}" for ext in ("svg", "pdf", "png", "tiff")}
        try:
            original = verify_table(root / expected["path"], expected)
            copied = verify_table(candidate_dir / f"figure_{number}_source_data.csv", expected)
            missing = [str(path) for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise ValueError(f"Missing or empty figure outputs: {missing}")
            svg = outputs["svg"].read_text(encoding="utf-8")
            with Image.open(outputs["png"]) as png:
                png.load()
                png_size = list(png.size)
            with Image.open(outputs["tiff"]) as tiff:
                tiff.load()
                tiff_size = list(tiff.size)
                tiff_dpi = [float(value) for value in tiff.info.get("dpi", (0, 0))]
            checks = {
                "source_hash_matches_frozen_manifest": original["sha256"] == expected["sha256"],
                "copied_hash_matches_frozen_manifest": copied["sha256"] == expected["sha256"],
                "source_and_copy_rows_match_manifest": original["rows"] == copied["rows"] == expected["rows"],
                "source_and_copy_bytes_match_manifest": original["bytes"] == copied["bytes"] == expected["bytes"],
                "all_formats_present": True,
                "svg_contains_editable_text": "<text" in svg,
                "pdf_signature_valid": outputs["pdf"].read_bytes()[:4] == b"%PDF",
                "png_nonempty": png_size[0] > 1000 and png_size[1] > 700,
                "tiff_approximately_600_dpi": len(tiff_dpi) == 2 and min(tiff_dpi) >= 590,
            }
            passed = all(checks.values())
            results.append({
                "figure": number, "status": "PASS" if passed else "FAIL",
                "expected_source_sha256": expected["sha256"], "source_sha256": original["sha256"],
                "source_rows": original["rows"], "checks": checks,
                "png_dimensions": png_size, "tiff_dimensions": tiff_size, "tiff_dpi": tiff_dpi,
                "output_sha256": {ext: sha256(path) for ext, path in outputs.items()},
            })
        except (OSError, ValueError) as exc:
            # Expected missing/corrupt-file failures are reported and fail the gate.
            errors.append(f"Figure {number}: {exc}")
            results.append({"figure": number, "status": "FAIL", "error": str(exc)})
    passed = not errors and len(results) == 6 and all(row["status"] == "PASS" for row in results)
    contacts = {}
    if passed:
        contacts = {
            "contact_sheet": str(contact_sheet(root).relative_to(root)),
            "grayscale_contact_sheet": str(contact_sheet(root, True).relative_to(root)),
        }
    return {
        "status": "AUTOMATED_PASS_VISUAL_REVIEW_PENDING" if passed else "FAIL",
        "automated_status": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "aggregate source integrity and figure file validation",
        "manual_visual_qa": {
            "status": "PENDING",
            "reason": "File checks cannot certify visual correctness; inspect the exported figures separately.",
        },
        "errors": errors, "figures": results, **contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit_figures(root)
    output = root / "outputs/figure_revision_qa_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    if report["automated_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
