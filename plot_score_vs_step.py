import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
parser.add_argument("--fig_name", type=str, default='plot.png')
args = parser.parse_args()

running_reward = []
num_of_steps = []

with open(args.file, 'r') as in_f:
    for line in in_f.readlines():
        if 'running_reward' in line:
            num_of_steps.append(int(line.split('step #')[1].split(',')[0]))
            running_reward.append(float(line.split('running_reward = ')[1].split(',')[0]))

assert (len(running_reward) == len(num_of_steps))

fig, ax = plt.subplots()
ax.scatter(num_of_steps, running_reward)

fig.savefig(args.fig_name)
