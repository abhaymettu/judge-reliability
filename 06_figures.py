"""Figures, from output/metrics.json only. No recomputation lives here.

Palette is the three validated categorical slots (blue, orange, aqua) used by
role, not by rank: blue is always an LLM judge, orange always a non LLM
baseline, aqua always a human reference line. The one exception is the mitigation
chart, where the same two hues mean "interval clears zero" and "does not", which
is a different job. Every bar carries its value as a direct label, so nothing is
encoded in colour alone.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "figures")

JUDGE = "#2a78d6"
BASELINE = "#eb6834"
HUMAN = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def role_colour(label):
    if "human" in label:
        return HUMAN
    if "baseline" in label:
        return BASELINE
    return JUDGE


def as_percent(axis, decimals=0):
    axis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def fig_agreement(m):
    rows = [r for r in m["agreement"]["no_ties"] if r["n"] > 0]
    rows.sort(key=lambda r: r["accuracy"])
    labels = [r["label"] for r in rows]
    values = [r["accuracy"] for r in rows]
    err = [[v - r["ci"][0] for v, r in zip(values, rows)], [r["ci"][1] - v for v, r in zip(values, rows)]]

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6))
    y = range(len(rows))
    ax.barh(y, values, height=0.62, color=[role_colour(l) for l in labels])
    ax.errorbar(values, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for i, (v, r) in enumerate(zip(values, rows)):
        ax.text(r["ci"][1] + 0.012, i, f"{v:.1%}  (n={r['n']})", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0.4, 1.16)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    as_percent(ax.xaxis)
    ax.set_xlabel("agreement with the human majority, ties excluded")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "A judge is only as good as the ceiling it is measured against",
        loc="left",
        fontsize=11,
        color=INK,
        pad=12,
    )
    finish(fig, os.path.join(OUT, "fig1_agreement.png"))


def fig_position(m):
    entries = [(k, v) for k, v in m["position_bias"].items() if isinstance(v, dict) and "flip_rate" in v]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(entries) + 2.0))
    labels = [k for k, _ in entries]
    values = [v["flip_rate"] for _, v in entries]
    err = [[v - e["flip_ci"][0] for v, (_, e) in zip(values, entries)],
           [e["flip_ci"][1] - v for v, (_, e) in zip(values, entries)]]
    y = range(len(entries))
    ax.barh(y, values, height=0.55, color=JUDGE)
    ax.errorbar(values, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for i, (v, (_, e)) in enumerate(zip(values, entries)):
        ax.text(e["flip_ci"][1] + 0.006, i, f"{v:.1%}   picks position one {e['position_one_rate']:.1%} of the time",
                va="center", fontsize=8, color=MUTED)
    rerun = m.get("rerun_stability")
    if rerun:
        noise = 1 - rerun["same_verdict_rate"]
        ax.axvline(noise, color=BASELINE, linewidth=1.6, linestyle="--")
        ax.text(noise, -0.85,
                f"  gpt-4 disagrees with itself on {noise:.1%} of identical\n  reruns of the same prompt, weeks apart",
                fontsize=8, color=BASELINE, va="center")
    ax.set_ylim(-1.4, len(entries) - 0.4)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, max(max(values), 0.3) * 1.55)
    as_percent(ax.xaxis)
    ax.set_xlabel("comparisons where reversing the presentation order reverses the verdict")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Position bias, against the judge's own run to run noise", loc="left", fontsize=11, color=INK, pad=12)
    finish(fig, os.path.join(OUT, "fig2_position_bias.png"))


def fig_verbosity(m):
    entries = [(k, v) for k, v in m["verbosity"].items() if v]
    entries.sort(key=lambda kv: kv[1]["picks_longer_rate"])
    fig, ax = plt.subplots(figsize=(8.2, 0.45 * len(entries) + 1.8))
    y = range(len(entries))
    values = [v["picks_longer_rate"] for _, v in entries]
    err = [[v - e["picks_longer_ci"][0] for v, (_, e) in zip(values, entries)],
           [e["picks_longer_ci"][1] - v for v, (_, e) in zip(values, entries)]]
    colours = [HUMAN if "human" in k else JUDGE for k, _ in entries]
    ax.barh(y, values, height=0.58, color=colours)
    ax.errorbar(values, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for i, (v, (_, e)) in enumerate(zip(values, entries)):
        ax.text(e["picks_longer_ci"][1] + 0.008, i, f"{v:.1%}   r={e['corr_choice_lengthdiff']:.2f}", va="center", fontsize=8, color=MUTED)
    ax.axvline(0.5, color=GRID, linewidth=1.0)
    ax.set_yticks(list(y), [k for k, _ in entries])
    ax.set_xlim(0.4, 1.05)
    as_percent(ax.xaxis)
    ax.set_xlabel("share of decided comparisons where the longer answer is chosen")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Everyone prefers the longer answer, humans included", loc="left", fontsize=11, color=INK, pad=12)
    finish(fig, os.path.join(OUT, "fig3_verbosity.png"))


def fig_prompt_sensitivity(m):
    """One line per judge prompt, all from the same model on the same comparisons.
    Swap averaged configurations are deliberately absent: this figure answers
    "what does the prompt alone do", and a mitigation would muddy that."""
    boards = m["leaderboards"]
    human = boards["human majority"]["win_rates"]
    prompts_shown = sorted(
        k for k in boards if k.startswith("local 3B") and k.endswith("one order")
    )
    if not prompts_shown:
        prompts_shown = [k for k in boards if k.startswith("gpt-4")]
    models = sorted(human, key=human.get, reverse=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = range(len(models))
    ax.plot(x, [human[m_] for m_ in models], marker="o", markersize=7, linewidth=2.4,
            color=HUMAN, label="human majority", zorder=5)
    for dash, key in zip(["-", "--", ":", "-."], prompts_shown):
        board = boards[key]["win_rates"]
        ax.plot(x, [board.get(m_, float("nan")) for m_ in models], marker="o", markersize=6,
                linewidth=1.7, linestyle=dash, color=JUDGE,
                label=key.replace("local 3B, ", "").replace(", one order", ""))
    ax.set_xticks(list(x), models, rotation=20, ha="right")
    as_percent(ax.yaxis)
    ax.set_ylabel("win rate over all comparisons")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("The same responses, ranked by judge prompt alone", loc="left", fontsize=11, color=INK, pad=12)
    finish(fig, os.path.join(OUT, "fig4_prompt_sensitivity.png"))


def fig_mitigations(m):
    rows = m["mitigations"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(rows) + 1.9))
    y = range(len(rows))
    deltas = [r["delta"] for r in rows]
    err = [[d - r["ci"][0] for d, r in zip(deltas, rows)], [r["ci"][1] - d for d, r in zip(deltas, rows)]]
    colours = [JUDGE if r["ci"][0] > 0 else BASELINE for r in rows]
    ax.barh(y, deltas, height=0.55, color=colours)
    ax.errorbar(deltas, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for i, (d, r) in enumerate(zip(deltas, rows)):
        side = 1 if d >= 0 else -1
        ax.text(r["ci"][1] + 0.004 if d >= 0 else r["ci"][0] - 0.004, i,
                f"{d:+.1%}", va="center", ha="left" if side > 0 else "right", fontsize=8, color=MUTED)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(list(y), [r["mitigation"] for r in rows])
    as_percent(ax.xaxis)
    ax.set_xlabel("change in agreement with the human majority")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    span = max(abs(r["ci"][0]) for r in rows), max(abs(r["ci"][1]) for r in rows)
    ax.set_xlim(-max(span) * 1.5, max(span) * 1.6)
    ax.set_title("Debiasing moves, measured rather than assumed", loc="left", fontsize=11, color=INK, pad=12)
    finish(fig, os.path.join(OUT, "fig5_mitigations.png"))


def fig_cost(m):
    """Reliability against price, with every point on ONE comparison pool.

    The local judge only ran on a subsample. Plotting its accuracy on the same
    axis as GPT-4's full set accuracy would compare two different samples as if
    they were one, so when the local judge is present the whole figure switches
    to the subsample both were measured on, and the title says so.
    """
    on_subsample = any(
        r["label"].startswith("local 3B") for r in m["agreement"]["no_ties"] if r["n"] > 0
    )
    source = m["agreement_on_local_subsample"] if on_subsample else m["agreement"]["no_ties"]
    acc = {r["label"]: r for r in source if r["n"] > 0}
    cost_by_judge = {c["judge_id"]: c for c in m["cost"]}
    points = []
    mapping = {
        "gpt-4 pairwise, one order": ("gpt-4-0613-pair", 1),
        "gpt-4 pairwise, swap averaged": ("gpt-4-0613-pair", 2),
        "gpt-4 single answer grading": ("gpt-4-0613-single", 2),
        "local 3B, rubric prompt, one order": ("qwen2.5-3b-4bit-local", 1),
        "local 3B, rubric prompt, swap averaged": ("qwen2.5-3b-4bit-local", 2),
    }
    for label, (judge_id, calls) in mapping.items():
        if label in acc and judge_id in cost_by_judge:
            usd = cost_by_judge[judge_id]["usd_per_1000_judgments"] * calls
            points.append((usd, acc[label]["accuracy"], label))
    for label in ("length baseline", "random baseline"):
        if label in acc:
            points.append((0.0, acc[label]["accuracy"], label))
    if not points:
        return

    local_speed = cost_by_judge.get("qwen2.5-3b-4bit-local", {}).get("median_latency_s")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # Nudge labels apart when two points nearly coincide, which they do at zero.
    placed = []
    for usd, a, label in sorted(points, key=lambda p: (p[0], -p[1])):
        colour = BASELINE if "baseline" in label else JUDGE
        ax.scatter([usd], [a], s=90, color=colour, zorder=4, edgecolor="white", linewidth=1.2)
        price = f"${usd:.0f} per 1000"
        if usd == 0 and label.startswith("local") and local_speed:
            price = f"no API cost, {local_speed:.1f} s per judgment on a laptop"
        offset = 9
        while any(abs(usd - u) < 45 and abs((a + offset / 400) - p) < 0.030 for u, p in placed):
            offset -= 15
        placed.append((usd, a + offset / 400))
        # A leader line, so a nudged label can never be read as belonging to the
        # neighbouring point. Wrong attribution is the expensive kind of mistake.
        ax.annotate(f"{label}\n{price}, {a:.1%}", (usd, a), textcoords="offset points",
                    xytext=(12, offset), fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.9,
                                    shrinkA=0, shrinkB=6))
    # Whichever human ceiling was computed on this pool. On the subsample only
    # the annotator against annotator one has enough votes to exist.
    ceiling = next((r for r in source if r["label"].startswith("human ceiling")), None) or next(
        (r for r in source if r["label"].startswith("human")), None
    )
    if ceiling:
        ax.axhline(ceiling["accuracy"], color=HUMAN, linewidth=1.6, linestyle="--")
        ax.text(0.995, ceiling["accuracy"] + 0.005, ceiling["label"] + "  ", color=HUMAN, fontsize=8,
                ha="right", transform=ax.get_yaxis_transform(which="grid"))
    ax.set_xlabel("USD per 1000 comparisons (estimated from cached token counts)")
    as_percent(ax.yaxis)
    ax.set_ylabel("agreement with the human majority")
    ax.set_xlim(-14, max(p[0] for p in points) * 1.75 + 8)
    ax.set_ylim(min(p[1] for p in points) - 0.09, max(max(p[1] for p in points), ceiling["accuracy"] if ceiling else 0) + 0.05)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    pool = (
        f"all judges on the {m['dataset']['local_judge_subsample']} comparison subsample"
        if on_subsample
        else f"all {m['dataset']['comparisons']} comparisons"
    )
    ax.set_title(f"Reliability against price, {pool}", loc="left", fontsize=11, color=INK, pad=12)
    finish(fig, os.path.join(OUT, "fig6_cost_reliability.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "output", "metrics.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    print("figures:")
    fig_agreement(m)
    fig_position(m)
    fig_verbosity(m)
    fig_prompt_sensitivity(m)
    fig_mitigations(m)
    fig_cost(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
