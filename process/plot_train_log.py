import re
import matplotlib.pyplot as plt
import scienceplots


def parse_log_file(filepath, skip=10):
    epochs = []
    losses = []
    rmse_E = []
    rmse_F = []
    rmse_stress = []
    lrs = []
    d3s = []

    with open(filepath, "r") as f:
        lines = f.readlines()[skip:]
        for line in lines:
            match = re.search(
                r"Epoch (\d+):.*?loss=([\d.eE+-]+), RMSE_E_per_atom=\s*([\d.]+) meV,"
                r"\s*RMSE_F=\s*([\d.]+) meV / A, RMSE_stress=\s*([\d.]+) meV / A\^3,"
                r"lr=([\d.eE+-]+),d3=([\d.]+)",
                line,
            )

            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                rmse_E.append(float(match.group(3)))
                rmse_F.append(float(match.group(4)))
                rmse_stress.append(float(match.group(5)))
                lrs.append(float(match.group(6)))
                d3s.append(float(match.group(7)))

    return epochs, losses, rmse_E, rmse_F, rmse_stress, lrs, d3s


def plot_all_metrics(epochs, losses, rmse_E, rmse_F, rmse_stress, lrs, d3s):
    with plt.style.context(["science", "grid"]):

        fig, axs = plt.subplots(4, 2, figsize=(16, 8))
        axs = axs.flatten()

        axs[0].plot(epochs, losses, marker="o")
        axs[0].set_title("Loss")
        axs[0].set_ylabel("Loss")

        axs[1].plot(epochs, rmse_E, marker="s")
        axs[1].set_title("RMSE_E_per_atom")
        axs[1].set_ylabel("meV")

        axs[2].plot(epochs, rmse_F, marker="^")
        axs[2].set_title("RMSE_F")
        axs[2].set_ylabel("meV / Å")

        axs[3].plot(epochs, rmse_stress, marker="d")
        axs[3].set_title("RMSE_stress")
        axs[3].set_ylabel("meV / Å³")

        axs[4].plot(epochs, lrs, marker="x")
        axs[4].set_title("Learning Rate")
        axs[4].set_ylabel("lr")

        axs[5].plot(epochs, d3s, marker="*")
        axs[5].set_title("d3")
        axs[5].set_ylabel("d3")

        # 空的子图隐藏掉
        for i in range(6, 8):
            axs[i].axis("off")

        for ax in axs[:6]:
            ax.set_xlabel("Epoch")
            ax.grid(True)

        plt.tight_layout()
        plt.show()


# 替换为你的日志文件路径
log_file_path = "/home/giga/code/msms/run/logs/pfpef_macecso3_run-1.log"
skip = 2821
data = parse_log_file(log_file_path, skip)
plot_all_metrics(*data)
