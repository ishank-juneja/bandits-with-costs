Inside a folder of bandit instances, the following are the steps:

1. If the folder/collection of bandit instances does not already exist, create it by using a suitable modification of
the script aaai_scripts/make_bandit_family.sh.
2. Run the algorithms to be compared and analyzed on every instance contained in the folder using the script
aaai_scripts/run_experiment_on_family.sh (or similar)
3. Once a separate log file for every instance in the family has been produced by running an identical experiment on
every instance in the family, create a scatter plot out of the results with quality regret on the x-axis and
cost regret on the y-axis. This is done using the pycharm run profile called "scatter_computer_generated"
