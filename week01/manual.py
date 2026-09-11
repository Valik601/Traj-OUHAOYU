import pandas as pd
import matplotlib.pyplot as plt

columns = [
    "frame", "agent",
    "x", "z", "y",
    "vx", "vz", "vy"
]

data = pd.read_csv(
    r"F:\OHYCode\Project\DATAPROCESSING\OpenTraj\datasets\UCY\students03\obsmat.txt",
    sep=r"\s+",
    header=None,
    names=columns
)


data = data.sort_values(["agent", "frame"])

for agent_id, track in data.groupby("agent"):
    plt.plot(
        track["x"],
        track["y"],
        linewidth=0.7,
        alpha=0.6
    )

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.title("students03 pedestrian trajectories")
plt.savefig("result_A1_students03.png")