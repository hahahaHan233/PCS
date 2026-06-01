from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

FIGURE_MAP = {
    ROOT / "output" / "modeling" / "roc_curve.png": ROOT / "latex_demo_project" / "figures" / "fig2_roc_curve.pdf",
    ROOT / "output" / "modeling" / "calibration_curve.png": ROOT / "latex_demo_project" / "figures" / "fig3_calibration_curve.pdf",
    ROOT / "output" / "modeling" / "dca_curve.png": ROOT / "latex_demo_project" / "figures" / "fig4_dca_curve.pdf",
    ROOT / "output" / "shap_analysis" / "shap_summary_plot.png": ROOT / "latex_demo_project" / "figures" / "fig5_shap_summary.pdf",
    ROOT / "output" / "shap_analysis" / "shap_variable_importance.png": ROOT / "latex_demo_project" / "figures" / "fig6_shap_variable_importance.pdf",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_class_distribution.png": ROOT / "latex_demo_project" / "figures" / "figS1_symptom_distribution.pdf",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_confusion_matrix.png": ROOT / "latex_demo_project" / "figures" / "figS2_symptom_confusion_matrix.pdf",
    ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_ovr_roc.png": ROOT / "latex_demo_project" / "figures" / "figS3_symptom_ovr_roc.pdf",
}


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


def main() -> None:
    missing = []
    for source, destination in FIGURE_MAP.items():
        if not source.exists():
            missing.append(str(source))
            continue
        raster_to_pdf(source, destination)
        print(f"saved {destination}")
    if missing:
        raise FileNotFoundError("Missing source figures:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
