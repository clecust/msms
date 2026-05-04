import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scienceplots

# =========================
# 0) 数据（只取 RMSE / 密度）
# =========================
rmse = {
    "ec": {
        "al": {"energy": 0.5977, "force": 68.0870},
        "ff": {"energy": 1.1592, "force": 91.9649},
    },
    "pc": {
        "al": {"energy": 0.4506, "force": 62.8647},
        "ff": {"energy": 1.0933, "force": 86.0652},
    },
    "lipf6-pc": {
        "al": {"energy": 0.4003, "force": 66.5454},
        "ff": {"energy": 0.5412, "force": 91.0367},
    },
    "lipf6-ec": {
        "al": {"energy": 0.4527, "force": 75.2627},
        "ff": {"energy": 0.8436, "force": 98.0984},
    },
}

density = {
    "ec": {"ff": 1.245, "al": 1.281, "exp": 1.32},
    "pc": {"ff": 1.151, "al": 1.186, "exp": 1.20},
}

# =========================
# 1) 单位
# =========================
E_UNIT = "meV/atom"
F_UNIT = "meV/Å"
RHO_UNIT = "g/cm$^3$"

# =========================
# 2) 样式/配色
# =========================
COL_FF = "tab:blue"
COL_AL = "tab:green"
COL_EXP = "black"

SYSTEMS_4 = ["ec", "pc", "lipf6-ec", "lipf6-pc"]
SYSTEMS_2 = ["ec", "pc"]

# —— 标签映射：EC/PC 全大写；LiPF6 中 6 下标；EC/PC 大写
SYS_LABEL = {
    "ec": "EC",
    "pc": "PC",
    "lipf6-ec": r"LiPF$_6$-EC",
    "lipf6-pc": r"LiPF$_6$-PC",
}

# —— 图例英文标注：ff/al 的物理含义（而不是缩写）
# ff：trained on classical/empirical force-field data
# al：trained on active-learning collected data
MODEL_LABEL = {
    "ff": "Trained on empirical force-field data",
    "al": "Trained on active-learning data",
    "exp": "Experiment",
}

# 不同体系之间的距离 > 体系内不同模型距离
group_gap = 1.8
w2 = 0.32
dx2 = 0.20
w3 = 0.26
dx3 = 0.30


def nice_ylim(data, pad=0.08):
    dmin, dmax = float(np.min(data)), float(np.max(data))
    dr = max(dmax - dmin, 1e-12)
    y0 = dmin - pad * dr
    if y0 <= 0:
        y0 = dmin * 0.90
    y1 = dmax + pad * dr
    return y0, y1


# =========================
# 3) 准备数据
# =========================
E_ff = np.array([rmse[s]["ff"]["energy"] for s in SYSTEMS_4])
E_al = np.array([rmse[s]["al"]["energy"] for s in SYSTEMS_4])
F_ff = np.array([rmse[s]["ff"]["force"] for s in SYSTEMS_4])
F_al = np.array([rmse[s]["al"]["force"] for s in SYSTEMS_4])

rho_ff = np.array([density[s]["ff"] for s in SYSTEMS_2])
rho_al = np.array([density[s]["al"] for s in SYSTEMS_2])
rho_exp = np.array([density[s]["exp"] for s in SYSTEMS_2])

x4 = np.arange(len(SYSTEMS_4)) * group_gap
x2 = np.arange(len(SYSTEMS_2)) * group_gap

x4_labels = [SYS_LABEL[s] for s in SYSTEMS_4]
x2_labels = [SYS_LABEL[s] for s in SYSTEMS_2]

# =========================
# 4) 画图（一行三图）
# =========================
with plt.style.context(["science"]):
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            # 让数学下标看起来更自然（LiPF$_6$）
            "mathtext.default": "regular",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(8, 2), dpi=250)
    for i, ax in enumerate(axes):
        delta =  0.05 if i==1 else 0
        ax.text(
            -0.26 - delta,
            1.0,
            f"({chr(ord('a') + i)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)

    # ---- 图1：Energy RMSE ----
    ax = axes[0]
    b_ff = ax.bar(
        x4 - dx2,
        E_ff,
        width=w2,
        color=COL_FF,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["ff"],
    )
    b_al = ax.bar(
        x4 + dx2,
        E_al,
        width=w2,
        color=COL_AL,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["al"],
    )
    ax.set_xticks(x4)
    ax.set_xticklabels(x4_labels)
    ax.set_ylabel(f"Energy RMSE ({E_UNIT})")
    ax.set_ylim(*nice_ylim(np.r_[E_ff, E_al]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # ---- 图2：Force RMSE ----
    ax = axes[1]
    ax.bar(
        x4 - dx2,
        F_ff,
        width=w2,
        color=COL_FF,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["ff"],
    )
    ax.bar(
        x4 + dx2,
        F_al,
        width=w2,
        color=COL_AL,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["al"],
    )
    ax.set_xticks(x4)
    ax.set_xticklabels(x4_labels)
    ax.set_ylabel(f"Force RMSE ({F_UNIT})")
    ax.set_ylim(*nice_ylim(np.r_[F_ff, F_al]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # ---- 图3：Density ----
    ax = axes[2]
    ax.bar(
        x2 - dx3,
        rho_ff,
        width=w3,
        color=COL_FF,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["ff"],
    )
    ax.bar(
        x2,
        rho_al,
        width=w3,
        color=COL_AL,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["al"],
    )
    ax.bar(
        x2 + dx3,
        rho_exp,
        width=w3,
        color=COL_EXP,
        edgecolor="0.2",
        linewidth=0.6,
        label=MODEL_LABEL["exp"],
    )
    ax.set_xticks(x2)
    ax.set_xticklabels(x2_labels)
    ax.set_ylabel(f"Density ({RHO_UNIT})")
    ax.set_ylim(*nice_ylim(np.r_[rho_ff, rho_al, rho_exp], pad=0.06))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # 只放一个总图例（避免重复）
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        handlelength=1.6,
        columnspacing=1.4,
    )

    plt.tight_layout()

    # 可选保存
    # fig.savefig("rmse_density_1x3_pretty.png",  bbox_inches="tight")
    fig.savefig("rmse_density_1x3_pretty.pdf", bbox_inches="tight")

    plt.show()
