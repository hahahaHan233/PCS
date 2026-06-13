from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

FIGURE_MAP = {
    ROOT / "output" / "modeling" / "roc_curve.png": "fig2_roc_curve",
    ROOT / "output" / "modeling" / "calibration_curve.png": "fig3_calibration_curve",
    ROOT / "output" / "modeling" / "dca_curve.png": "fig4_dca_curve",
    ROOT / "output" / "shap_analysis" / "shap_summary_plot.png": "fig5_shap_summary",
    ROOT / "output" / "shap_analysis" / "shap_variable_importance.png": "fig6_shap_variable_importance",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_class_distribution.png": "figS1_symptom_distribution",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_confusion_matrix.png": "figS2_symptom_confusion_matrix",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_ovr_roc.png": "figS3_symptom_ovr_roc",
}
LATEX_FIGURES = ROOT / "latex_demo_project" / "figures"


def raster_to_pdf(source: Path, destination: Path) -> None:
    image = mpimg.imread(source)
    height, width = image.shape[:2]
    fig_width = 7.2
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def export_figure_set(source: Path, stem: str) -> None:
    for subdir in ("pdf", "png", "tiff"):
        (LATEX_FIGURES / subdir).mkdir(parents=True, exist_ok=True)
    raster_to_pdf(source, LATEX_FIGURES / "pdf" / f"{stem}.pdf")
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgb.save(LATEX_FIGURES / "png" / f"{stem}.png", "PNG", optimize=True)
        rgb.save(LATEX_FIGURES / "tiff" / f"{stem}.tiff", "TIFF", compression="tiff_lzw")


def main() -> None:
    missing = []
    for source, stem in FIGURE_MAP.items():
        if not source.exists():
            missing.append(str(source))
            continue
        export_figure_set(source, stem)
        print(f"saved {stem}")
    if missing:
        raise FileNotFoundError("Missing source figures:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
