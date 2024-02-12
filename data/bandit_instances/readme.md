Bandit Instances in this folder are MAB-CS (Multi-Armed Bandits with Cost Subsidy Instances)

The following fields are available
- instance_id[string]: unique identifier for the instance [Required]
- arm_reward_array[array]: Comma separated list of rewards for each arm [Required]
- arm_cost_array[array]: If specified it is Comma separated list of costs for each arm [Optional]
- min_reward[float]: If specified it is the minimum tolerated reward that quality regret is calibrated against [Optional]
- ref_arm_ell[int]: If specified it is the arm index that quality regret is calibrated against [Optional]
- subsidy_factor[float]: If specified it is the subsidy factor as described in the Multi-Armed Bandits with Cost Subsidy Paper [Optional]
