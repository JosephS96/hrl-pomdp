import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

import pandas as pd
import seaborn as sns

BASE_PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"

def create_dataframe(dict_values):
    pass

if __name__ == "__main__":
    env_name = 'MiniGrid-Empty-16x16-v0'
    agent_name = 'dqn'

    # file_path = f"{BASE_PATH}/{env_name}/{agent_name}-{time.time()}.npy"

    file_path = "/Users/josesanchez/Documents/IAS/Thesis-Results/FourRooms-13x13/double-dqn-1655219414.068259.npy"
    values = np.load(file_path, allow_pickle=True).item()

    data = pd.DataFrame.from_dict(values)

    print(datetime.now())

    plt.figure(figsize=(15, 8))
    sns.set(style='ticks')
    sns.set_style("darkgrid")
    plot = sns.lineplot(x=range(len(values.get('success'))), y=values.get('success'))
    plot.set_xlabel("Episodes", fontsize=10)
    plot.set_ylabel("Success rate", fontsize=10)
    plt.ylim(0, 1.0)
    plt.show()

    plt.figure(figsize=(15, 8))
    sns.set(style='ticks')
    sns.set_style("darkgrid")
    plot = sns.lineplot(x=range(len(values.get('rewards'))), y=values.get('rewards'))
    plot.set_xlabel("Rewards", fontsize=10)
    plot.set_ylabel("Episodes", fontsize=10)
    plt.ylim(0, 1.0)
    plt.show()