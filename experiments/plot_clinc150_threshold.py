import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(19680801)


def is_close_to_set(x, s):
    for el in s:
        if np.isclose(x, el):
            return True
    return False


def plot_bar(ax, data, title):
    colors = ["cornflowerblue", "lightsteelblue"]
    labels = ["0.4", "0.5", "0.6", "0.7", "0.8"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    ticks = np.arange(0, 1 + 0.1, 0.1)
    ax.yaxis.set_ticks(ticks)
    Y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticklabels([np.round(i, 1) if is_close_to_set(i, Y) else "" for i in ticks])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5)
    rects1 = ax.bar(x - width / 2, data[0], width, label="is", color=colors[0])
    rects2 = ax.bar(x + width / 2, data[1], width, label="oos", color=colors[1])

    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend(loc="lower right")

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()


acc0 = np.array(
    [[0.912, 0.33], [0.904, 0.660], [0.864, 0.88], [0.760, 0.966], [0.592, 0.993]]
)

data_us_mini = [[0.88, 0.87, 0.829, 0.728, 0.551], [0.41, 0.71, 0.87, 0.97, 0.99]]
data_us_rob = [[0.913, 0.903, 0.855, 0.747, 0.569], [0.49, 0.76, 0.9, 0.966, 0.99]]
data_s_mini = [[0.938, 0.935, 0.914, 0.9096, 0.8773], [0.38, 0.52, 0.63, 0.73, 0.87]]
data_s_rob = [[0.960, 0.953, 0.944, 0.9356, 0.9186], [0.53, 0.65, 0.75, 0.84, 0.89]]


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

plot_bar(ax0, data_us_mini, title="unsupervised, all-MiniLM-L6-v2")
plot_bar(ax1, data_s_mini, title="supervised, all-MiniLM-L6-v2")
plot_bar(ax2, data_us_rob, title="unsupervised, all-roberta-large-v1")
plot_bar(ax3, data_s_rob, title="supervised, all-roberta-large-v1")

path = Path(__file__).parent / "plots"
if not path.exists():
    path.mkdir(parents=True)

plt.savefig(str(path / "threshold.eps"), format="eps")
plt.show()
