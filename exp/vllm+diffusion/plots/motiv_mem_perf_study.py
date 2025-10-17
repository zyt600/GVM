#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, ScalarFormatter

# =========================
# ---- Input Metrics ------
# =========================
VLLM_P99_TTFT_MS = [246.42, 2551.86, 253.41, 9407.46]
VLLM_P99_ITL_MS  = [64.89,   174.60,  68.19,   994.65]
DIFFUSION_THROUGHPUT = [13.92, 7.4, 3.26, 1.17]  # req/min

CASE_DESCRIPTIONS = [
    "vLLM / Diffusion running exclusively",
    "Default GPU sharing (no mem pressure)",
    "GPU sharing w/ XSched (no mem pressure)",
    "GPU sharing w/ XSched (with mem pressure)",
]
CASE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]  # blue, orange, green, purple
HATCH_PATTERNS = ["/", "\\", "x", "."]

# =========================
# ---- Memory Plot I/O ----
# =========================
CSV_PATH = Path("memory_current_log.csv")
PLOT_TMAX = 175
STAGES_S: List[Tuple[float, float]] = [(0, 26), (26, 50), (50, 175)]
STAGE_COLORS = ["#b3d9e6", "#ffd699", "#99cc99"]

# =========================
# ---- Styling ------------
# =========================
LABELPAD = 18
YLABEL_XCOORD = -0.15
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,           
    "hatch.linewidth": 0.9,
    "figure.constrained_layout.use": True,
})

# =========================
# ---- Helpers ------------
# =========================
def _apply_hatches_to_bars(bars):
    for i, b in enumerate(bars):
        b.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

def put_panel_label(ax, text: str):
    ax.text(0.5, -0.18, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=26, fontweight="bold")

def add_case_legend(fig):
    handles = [
        Rectangle((0, 0), 1, 1,
                  facecolor=CASE_COLORS[i],
                  edgecolor="black",
                  linewidth=0.8,
                  hatch=HATCH_PATTERNS[i % len(HATCH_PATTERNS)])
        for i in range(4)
    ]
    fig.legend(
        handles, CASE_DESCRIPTIONS,
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=24,                  
    )

def _slash_marks_between(ax_top, ax_bot, size=0.02):
    kw = dict(color="k", clip_on=False, linewidth=1.5)
    ax_top.plot((-size, +size), (-size, +size), transform=ax_top.transAxes, **kw)
    ax_top.plot((1-size, 1+size), (-size, +size), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-size, +size), (1-size, 1+size), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1-size, 1+size), (1-size, 1+size), transform=ax_bot.transAxes, **kw)

def _pin_break_ticks(ax_top, ax_bot, y_low_max, y_high_min):
    # Bottom axis ticks
    bot_ticks = np.array(ax_bot.get_yticks(), dtype=float)
    bot_ticks = np.unique(np.append(bot_ticks[(bot_ticks >= 0) & (bot_ticks <= y_low_max)], y_low_max))
    ax_bot.yaxis.set_major_locator(FixedLocator(bot_ticks))

    # Top axis ticks
    top_ticks = np.array(ax_top.get_yticks(), dtype=float)
    top_ticks = np.unique(np.append(top_ticks[(top_ticks >= y_high_min)], y_high_min))
    ax_top.yaxis.set_major_locator(FixedLocator(top_ticks))

    # Clean numeric formatting
    for a in (ax_top, ax_bot):
        fmt = ScalarFormatter(useOffset=False, useMathText=False)
        fmt.set_scientific(False)
        a.yaxis.set_major_formatter(fmt)

def broken_axis_bar(
    ax, data, ylabel,
    y_low_max=None, y_high_min=None, y_high_max=None,
    *, keep_idx=(0, 2), lower_pad=0.18, upper_pad=0.08,
    height_ratios=(1, 1), add_break_numbers=True
):
    """Broken y-axis bar chart with one centered y-label aligned across the grid."""
    vals = np.asarray(data, float)
    n = len(vals); x = np.arange(n); colors = CASE_COLORS
    kept = np.array(keep_idx, int)
    comp = np.array([i for i in range(n) if i not in kept], int)

    # Auto ranges if not provided
    if y_low_max is None:
        y_low_max = (1 + lower_pad) * float(vals[kept].max())
    if y_high_min is None:
        base = float(vals[comp].min()) if comp.size else float(vals.max())
        y_high_min = max((1 - upper_pad) * base, y_low_max * 1.05)
    if y_high_max is None:
        y_high_max = (1 + upper_pad) * float(vals.max())

    fig = ax.figure
    subgs = ax.get_subplotspec().subgridspec(2, 1, height_ratios=height_ratios, hspace=0.04)
    ax.remove()

    # Bottom panel
    ax_bot = fig.add_subplot(subgs[1])
    bars_bot = ax_bot.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
    _apply_hatches_to_bars(bars_bot)
    ax_bot.set_ylim(0, y_low_max)
    ax_bot.set_xlim(-0.5, n - 0.5)
    ax_bot.set_xticks([])
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.set_ylabel("")

    # Top panel
    ax_top = fig.add_subplot(subgs[0], sharex=ax_bot)
    bars_top = ax_top.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
    _apply_hatches_to_bars(bars_top)
    ax_top.set_ylim(y_high_min, y_high_max)
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_top.set_xticks([])
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.set_ylabel("")

    _slash_marks_between(ax_top, ax_bot, size=0.02)

    # Numeric labels at the break without disturbing limits
    if add_break_numbers:
        _pin_break_ticks(ax_top, ax_bot, y_low_max, y_high_min)

    # Spanning axis to carry a single centered label
    ax_span = fig.add_subplot(subgs[:, 0], frameon=False)
    ax_span.set_xticks([]); ax_span.set_yticks([])
    for spine in ax_span.spines.values():
        spine.set_visible(False)
    ax_span.set_ylabel(ylabel, labelpad=LABELPAD)
    ax_span.yaxis.set_label_coords(YLABEL_XCOORD, 0.5)

    return ax_top, ax_bot, ax_span

def bar_simple(ax, data, ylabel):
    x = np.arange(len(data))
    bars = ax.bar(x, data, color=CASE_COLORS, edgecolor="black", linewidth=0.6)
    _apply_hatches_to_bars(bars)
    ax.set_ylabel(ylabel, labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD, 0.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

def load_memory_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce")
    t0 = df["timestamp"].dropna().iloc[0]
    df["time_s"] = (df["timestamp"] - t0).dt.total_seconds()
    df["memory_current_bytes"] = pd.to_numeric(df["memory_current_bytes"], errors="coerce")
    df["memory_gb"] = df["memory_current_bytes"] / (1024**3)
    return df

def plot_memory(ax):
    df = load_memory_df(CSV_PATH)
    dfp = df[df["time_s"] <= PLOT_TMAX].copy()
    engine = dfp[dfp["cmd"].str.contains("VLLM::EngineCore", na=False)]
    diff   = dfp[dfp["cmd"].str.contains("diffusion.py",    na=False)]

    for i, (start, end) in enumerate(STAGES_S, start=1):
        ax.axvspan(start, end, facecolor=STAGE_COLORS[(i-1) % len(STAGE_COLORS)], alpha=0.35, zorder=0)
        ax.axvline(start, linestyle="--", alpha=0.35, zorder=1)
        ax.axvline(end,   linestyle="--", alpha=0.35, zorder=1)
        x_mid = 0.5 * (start + end)
        ax.text(x_mid, 0.98, f"#{i}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=20, fontweight="bold", color="black", zorder=2)

    ax.plot(engine["time_s"], engine["memory_gb"], label="vLLM", linewidth=2)
    ax.plot(diff["time_s"],   diff["memory_gb"],   label="Diffusion", linewidth=2)

    ymax = max(engine["memory_gb"].max() if not engine.empty else 0.0,
               diff["memory_gb"].max() if not diff.empty else 0.0)
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(bottom=0, top=max(ymax * 1.3, 40))
    else:
        ax.set_ylim(bottom=0, top=40)

    ax.axhline(40, color="red", linestyle="--", linewidth=2, label="Hardware Limit")

    ax.set_xlim(0, PLOT_TMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Memory Usage (GB)", labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD, 0.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.legend(loc="center", bbox_to_anchor=(0.75, 0.45), frameon=False, fontsize=24)

def main():
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_ttft  = fig.add_subplot(gs[0, 0])
    ax_itl   = fig.add_subplot(gs[0, 1])
    ax_tput  = fig.add_subplot(gs[1, 0])
    ax_mem   = fig.add_subplot(gs[1, 1])

    # vLLM with broken axes
    ttft_top, ttft_bot, _ = broken_axis_bar(
        ax_ttft, VLLM_P99_TTFT_MS, "vLLM P99 TTFT (ms)",
        y_low_max=300, y_high_min=1000, y_high_max=12000,
        keep_idx=(0, 2), height_ratios=(1, 1), add_break_numbers=True,
    )
    put_panel_label(ttft_bot, "(a)")

    itl_top, itl_bot, _ = broken_axis_bar(
        ax_itl, VLLM_P99_ITL_MS, "vLLM P99 ITL (ms)",
        y_low_max=90, y_high_min=120, y_high_max=1200,
        keep_idx=(0, 2), height_ratios=(1, 1), add_break_numbers=True,
    )
    put_panel_label(itl_bot, "(b)")

    # Diffusion throughput
    bar_simple(ax_tput, DIFFUSION_THROUGHPUT, ylabel="Diffusion Tput (req/min)")
    put_panel_label(ax_tput, "(c)")

    # Memory usage
    if CSV_PATH.exists():
        plot_memory(ax_mem)
    else:
        ax_mem.text(0.5, 0.5, f"Missing {CSV_PATH.name}", ha="center", va="center", fontsize=18)
        ax_mem.set_axis_off()
    put_panel_label(ax_mem, "(d)")

    add_case_legend(fig)

    out = Path("motiv_mem_perf_study.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()

if __name__ == "__main__":
    main()
