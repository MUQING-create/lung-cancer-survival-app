"""Regenerate the README model-evaluation figures from the deployment bundle."""

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family'      : 'Arial',
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 9,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8,
    'axes.spines.top'  : True,
    'axes.spines.right': True,
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'pdf.fonttype'     : 42,   # embeds fonts as TrueType in PDF
    'ps.fonttype'      : 42,
})

CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

PRIMARY_COLOR = CELL_COLORS[0]
SECONDARY_COLOR = CELL_COLORS[1]
TERTIARY_COLOR = CELL_COLORS[2]
TEXT_COLOR = CELL_COLORS[3]
WARNING_COLOR = CELL_COLORS[4]
REFERENCE_COLOR = CELL_COLORS[9]
FIT_COLORS = [SECONDARY_COLOR, TERTIARY_COLOR, TEXT_COLOR, WARNING_COLOR]

mpl.rcParams.update({
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "tcga_luad_app_bundle.pkl"
ASSET_DIR = ROOT / "assets"


def _load_bundle() -> dict:
    with BUNDLE_PATH.open("rb") as handle:
        return pickle.load(handle)


def _style_axis(ax) -> None:
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TEXT_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(
        direction="out",
        length=3,
        width=0.8,
        colors=TEXT_COLOR,
    )
    ax.grid(False)


def _save_figure(fig, filename: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / filename, format="png", dpi=300)
    plt.close(fig)


def _model_performance_figure(bundle: dict) -> None:
    model_rows = [
        {
            "name": "Cox PH",
            "color": PRIMARY_COLOR,
            "train": bundle["tr_cox"],
            "test": bundle["res_cox"],
        },
        {
            "name": f"{bundle['dist']['best']} AFT",
            "color": SECONDARY_COLOR,
            "train": bundle["tr_aft"],
            "test": bundle["res_aft"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.5), dpi=300)
    for ax in axes:
        _style_axis(ax)

    ax = axes[0]
    y_positions = np.arange(len(model_rows))[::-1]
    ax.axvline(0.5, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
    for y_pos, row in zip(y_positions, model_rows):
        test = row["test"]
        test_value = float(test["c_index"])
        ax.plot(
            float(row["train"]["c_index"]),
            y_pos + 0.10,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=row["color"],
            markeredgewidth=1.0,
            linestyle="none",
        )
        ax.errorbar(
            test_value,
            y_pos - 0.10,
            xerr=np.array(
                [[test_value - float(test["ci_lo"])],
                 [float(test["ci_hi"]) - test_value]]
            ),
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            markersize=4,
            elinewidth=0.8,
            capsize=3,
        )
        ax.text(
            min(float(test["ci_hi"]) + 0.008, 0.825),
            y_pos - 0.10,
            f"{test_value:.3f}",
            va="center",
            fontsize=7,
            color=row["color"],
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Cox PH", "Log-Logistic\nAFT"])
    ax.set_xlim(0.48, 0.88)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Concordance index")
    ax.set_title("A. Discrimination", loc="left")
    ax.text(
        0.02,
        0.04,
        "Open marker = train\nFilled marker = test (95% CI)",
        transform=ax.transAxes,
        fontsize=6.2,
    )

    ax = axes[1]
    ibs_values = []
    for y_pos, row in zip(y_positions, model_rows):
        train_value = float(row["train"]["ibs"])
        test_value = float(row["test"]["ibs"])
        ibs_values.extend([train_value, test_value])
        ax.plot(
            [train_value, test_value],
            [y_pos + 0.10, y_pos - 0.10],
            color=REFERENCE_COLOR,
            linewidth=1.0,
        )
        ax.plot(
            train_value,
            y_pos + 0.10,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=row["color"],
            markeredgewidth=1.0,
            linestyle="none",
        )
        ax.plot(
            test_value,
            y_pos - 0.10,
            marker="o",
            markersize=4,
            color=row["color"],
            linestyle="none",
        )
        ax.text(
            test_value + 0.00025,
            y_pos - 0.10,
            f"{test_value:.3f}",
            va="center",
            fontsize=7,
            color=row["color"],
        )
    padding = max(ibs_values) - min(ibs_values)
    padding = max(padding * 0.8, 0.0015)
    ax.set_xlim(min(ibs_values) - padding, max(ibs_values) + padding * 1.5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Cox PH", "Log-Logistic\nAFT"])
    ax.set_xlabel("Integrated Brier score")
    ax.set_title("B. Overall prediction error", loc="left")
    ax.text(
        0.98,
        0.04,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.6,
    )

    ax = axes[2]
    for index, row in enumerate(model_rows):
        test = row["test"]
        times = np.asarray(test["times"], dtype=float)
        auc_values = np.asarray(test["auc_vals"], dtype=float)
        ax.plot(
            times,
            auc_values,
            color=row["color"],
            linewidth=1.0,
            linestyle="-" if index == 0 else "--",
            marker="o",
            markersize=4,
            markerfacecolor=row["color"] if index == 0 else "white",
            markeredgecolor=row["color"],
        )
        ax.text(
            66.0,
            float(auc_values[-1]),
            "Cox" if index == 0 else "AFT",
            va="center",
            ha="right",
            fontsize=6.4,
            color=row["color"],
        )
    ax.axhline(0.5, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xlim(10, 67)
    ax.set_xticks([12, 24, 36, 48, 60])
    ax.set_ylim(0.48, 0.84)
    ax.set_xlabel("Time since diagnosis (months)")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title("C. Time-dependent discrimination", loc="left")
    ax.text(
        0.02,
        0.09,
        "Mean AUC: Cox 0.748; AFT 0.758",
        transform=ax.transAxes,
        fontsize=6.2,
    )

    _save_figure(fig, "model_performance.png")


def _marginal_survival_figure(bundle: dict) -> None:
    distribution = bundle["dist"]
    km_fitter = bundle["km_train"]
    fitters = distribution["fitters"]
    best_name = distribution["best"]
    ordered_names = [best_name] + [name for name in fitters if name != best_name]
    time_grid = np.linspace(0.0, 180.0, 500)

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
    _style_axis(ax)

    km_curve = km_fitter.survival_function_.iloc[:, 0]
    ax.step(
        km_curve.index.to_numpy(dtype=float),
        km_curve.to_numpy(dtype=float),
        where="post",
        color=PRIMARY_COLOR,
        linewidth=1.0,
        label="Kaplan-Meier",
    )

    line_styles = ["-", "--", "-.", ":"]
    for index, (name, line_style) in enumerate(zip(ordered_names, line_styles)):
        survival = np.asarray(
            fitters[name].survival_function_at_times(time_grid),
            dtype=float,
        ).reshape(-1)
        label = f"{name} (best)" if name == best_name else name
        ax.plot(
            time_grid,
            survival,
            color=FIT_COLORS[index],
            linewidth=1.0,
            linestyle=line_style,
            label=label,
        )

    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time since diagnosis (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Marginal survival fit", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.4)

    _save_figure(fig, "marginal_survival_fit.png")


def main() -> None:
    bundle = _load_bundle()
    _model_performance_figure(bundle)
    _marginal_survival_figure(bundle)
    print("README figures regenerated with the shared Cell color system.")


if __name__ == "__main__":
    main()
