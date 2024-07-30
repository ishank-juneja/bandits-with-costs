import pandas as pd
import numpy as np
import argparse
import pathlib

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("--log-file", action="store", dest="file")
parser.add_argument('--save-dir', type=str, help='The directory to save the plots in.')
args = parser.parse_args()

in_file = args.file
in_name = in_file.split('/')[-1].split('.')[0][:-4]

bandit_data = pd.read_csv(in_file, sep=",")
horizon = bandit_data["time-step"].max()
plot_step = bandit_data["time-step"].iloc[1] - bandit_data["time-step"].iloc[0]
bandit_data['nsamps'] = bandit_data['nsamps'].apply(lambda x: np.fromstring(x, dtype=int, sep=';'))

# Same x-points used for all the plots
x_points = np.arange(0, horizon + 1, plot_step)


# Filter out the data corresponding to just this algorithm
algo_data = bandit_data[bandit_data["algo"] == 'cs-ts']
# Assuming 'horizon' is defined and 'bandit_data' is your DataFrame
# Filter the data for the specific algorithm and at the maximum time-step
algo_horizon_data = algo_data[algo_data["time-step"] == horizon]

# Sort the data by 'qual_reg' in descending order to get the top performances
sorted_data = algo_horizon_data.sort_values('qual_reg', ascending=False)

# Display the qual regret for the top 10 seeds and the seeds in a tabular format
# Making sure all 100 rows are printed
pd.set_option('display.max_rows', 100)
print(sorted_data[['rs', 'qual_reg']].head(100))
