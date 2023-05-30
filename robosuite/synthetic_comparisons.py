import numpy as np
from robosuite.environments.manipulation.lift_features import gt_reward, speed, height, distance_to_bottle, distance_to_cube


greater_gtreward_adjs = ["better", "more successfully"]
less_gtreward_adjs = ["worse", "not as well"]
greater_speed_adjs = ["faster", "quicker", "swifter", "at a higher speed"]
less_speed_adjs = ["slower", "more moderate", "more sluggish", "at a lower speed"]
greater_height_adjs = ["higher", "taller", "at a greater height"]
less_height_adjs = ["lower", "shorter", "at a lesser height"]
greater_distance_adjs = ["further", "farther", "more distant"]
less_distance_adjs = ["closer", "nearer", "more nearby"]


GT_REWARD_MEAN = None
GT_REWARD_STD = None
SPEED_MEAN = None
SPEED_STD = None
HEIGHT_MEAN = None
HEIGHT_STD = None
DISTANCE_TO_BOTTLE_MEAN = None
DISTANCE_TO_BOTTLE_STD = None
DISTANCE_TO_CUBE_MEAN = None
DISTANCE_TO_CUBE_STD = None


def calc_and_set_global_vars(trajs):
    horizon = len(trajs[0])
    avg_gt_rewards = []
    avg_speeds = []
    avg_heights = []
    avg_distance_to_bottles = []
    avg_distance_to_cubes = []

    for traj in trajs:
        avg_gt_rewards.append(np.mean([gt_reward(traj[t]) for t in range(horizon)]))
        avg_speeds.append(np.mean([speed(traj[t]) for t in range(horizon)]))
        avg_heights.append(np.mean([height(traj[t]) for t in range(horizon)]))
        avg_distance_to_bottles.append(np.mean([distance_to_bottle(traj[t]) for t in range(horizon)]))
        avg_distance_to_cubes.append(np.mean([distance_to_cube(traj[t]) for t in range(horizon)]))

    global GT_REWARD_MEAN
    global GT_REWARD_STD
    global SPEED_MEAN
    global SPEED_STD
    global HEIGHT_MEAN
    global HEIGHT_STD
    global DISTANCE_TO_BOTTLE_MEAN
    global DISTANCE_TO_BOTTLE_STD
    global DISTANCE_TO_CUBE_MEAN
    global DISTANCE_TO_CUBE_STD

    GT_REWARD_MEAN = np.mean(avg_gt_rewards)
    GT_REWARD_STD = np.std(avg_gt_rewards)
    SPEED_MEAN = np.mean(avg_speeds)
    SPEED_STD = np.std(avg_speeds)
    HEIGHT_MEAN = np.mean(avg_heights)
    HEIGHT_STD = np.std(avg_speeds)
    DISTANCE_TO_BOTTLE_MEAN = np.mean(avg_distance_to_bottles)
    DISTANCE_TO_BOTTLE_STD = np.std(avg_speeds)
    DISTANCE_TO_CUBE_MEAN = np.mean(avg_distance_to_cubes)
    DISTANCE_TO_CUBE_STD = np.std(avg_speeds)


# This function takes in two trajectories in the form of LISTS of (observation, action) pairs.
# Features:
# GT reward (mean)
# Speed (mean)
# Height (mean)
# Distance to cube (final)
# Distance to bottle (min)
def generate_synthetic_comparisons_statements(traj1, traj2, feature_name):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "gt_reward":
        traj1_feature_values = [gt_reward(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [gt_reward(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN gt_reward
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in greater_gtreward_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in less_gtreward_adjs]
            return ordinary_comps + flipped_comps
        else:
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in less_gtreward_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in greater_gtreward_adjs]
            return ordinary_comps + flipped_comps

    elif feature_name == "speed":
        traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN speed
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in greater_speed_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in less_speed_adjs]
            return ordinary_comps + flipped_comps
        else:
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in less_speed_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in greater_speed_adjs]
            return ordinary_comps + flipped_comps

    elif feature_name == "height":
        traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [height(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN height
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in greater_height_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in less_height_adjs]
            return ordinary_comps + flipped_comps
        else:
            ordinary_comps = ["The first trajectory is " + w + " than the second trajectory." for w in less_height_adjs]
            flipped_comps = ["The second trajectory is " + w + " than the first trajectory." for w in greater_height_adjs]
            return ordinary_comps + flipped_comps

    elif feature_name == "distance_to_bottle":
        traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]

        # TODO: Later, we can make this non-Markovian (e.g., the MINIMUM distance)
        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN distance
            ordinary_comps = ["The first trajectory is " + w + " from the bottle than the second trajectory." for w in greater_distance_adjs]
            flipped_comps = ["The second trajectory is " + w + " to the botte than the first trajectory." for w in less_distance_adjs]
            return ordinary_comps + flipped_comps
        else:
            # The first trajectory is further from the bottle than the second trajectory.
            ordinary_comps = ["The first trajectory is " + w + " to the bottle than the second trajectory." for w in less_distance_adjs]
            flipped_comps = ["The second trajectory is " + w + " from the bottle than the first trajectory." for w in greater_distance_adjs]
            return ordinary_comps + flipped_comps

    elif feature_name == "distance_to_cube":
        traj1_feature_values = [distance_to_cube(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_cube(traj2[t]) for t in range(horizon)]

        # TODO: Later, we can make this non-Markovian (e.g., the FINAL distance)
        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN distance
            ordinary_comps = ["The first trajectory is " + w + " from the cube than the second trajectory." for w in greater_distance_adjs]
            flipped_comps = ["The second trajectory is " + w + " to the cube than the first trajectory." for w in less_distance_adjs]
            return ordinary_comps + flipped_comps
        else:
            ordinary_comps = ["The first trajectory is " + w + " to the cube than the second trajectory." for w in less_distance_adjs]
            flipped_comps = ["The second trajectory is " + w + " from the cube than the first trajectory." for w in greater_distance_adjs]
            return ordinary_comps + flipped_comps


# NOTE: For this function, we produce commands that would change traj1 to traj2.
def generate_synthetic_comparisons_commands(traj1, traj2, feature_name):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "gt_reward":
        traj1_feature_values = [gt_reward(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [gt_reward(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN gt_reward
            commands = ["Lift the cube " + w + "." for w in greater_gtreward_adjs]
            return commands
        else:
            commands = ["Lift the cube " + w + "." for w in less_gtreward_adjs]
            return commands

    elif feature_name == "speed":
        traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN speed
            commands = ["Move " + w + "." for w in greater_speed_adjs]
            return commands
        else:
            commands = ["Move " + w + "." for w in less_speed_adjs]
            return commands

    elif feature_name == "height":
        traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [height(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN height
            commands = ["Move " + w + "." for w in greater_height_adjs]
            return commands
        else:
            commands = ["Move " + w + "." for w in less_height_adjs]
            return commands

    elif feature_name == "distance_to_bottle":
        traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]

        # TODO: Later, we can make this non-Markovian (e.g., the MINIMUM distance)
        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN distance
            commands = ["Move " + w + " from the bottle." for w in greater_distance_adjs]
            return commands
        else:
            # Move further from the bottle.
            commands = ["Move " + w + " to the bottle." for w in less_distance_adjs]
            return commands

    elif feature_name == "distance_to_cube":
        traj1_feature_values = [distance_to_cube(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_cube(traj2[t]) for t in range(horizon)]

        # TODO: Later, we can make this non-Markovian (e.g., the FINAL distance)
        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN distance
            commands = ["Move " + w + " from the cube." for w in greater_distance_adjs]
            return commands
        else:
            commands = ["Move " + w + " to the cube." for w in less_distance_adjs]
            return commands


# NOTE: For this function, we produce commands that would change traj1 to traj2.
# We generate n comparisons per trajectory pair, where the proportion that are greaterly labeled
# is equal to the sigmoid of the difference in the feature values.
# IMPORTANT: User needs to have run calc_and_set_global_vars() at some point before this function.
def generate_noisyaugmented_synthetic_comparisons_commands(traj1, traj2, feature_name, n_duplicates):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "gt_reward":
        traj1_feature_values = [gt_reward(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [gt_reward(traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff/GT_REWARD_STD))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            commands.extend(["Lift the cube " + w + "." for w in greater_gtreward_adjs])
        for i in range(num_lesser):
            commands.extend(["Lift the cube " + w + "." for w in less_gtreward_adjs])

        return commands

    elif feature_name == "speed":
        traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff/SPEED_STD))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            commands.extend(["Move " + w + "." for w in greater_speed_adjs])
        for i in range(num_lesser):
            commands.extend(["Move " + w + "." for w in less_speed_adjs])

        return commands

    elif feature_name == "height":
        traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [height(traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff/HEIGHT_STD))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            commands.extend(["Move " + w + "." for w in greater_height_adjs])
        for i in range(num_lesser):
            commands.extend(["Move " + w + "." for w in less_height_adjs])

        return commands

    elif feature_name == "distance_to_bottle":
        traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff/DISTANCE_TO_BOTTLE_STD))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            commands.extend(["Move " + w + " from the bottle." for w in greater_distance_adjs])
        for i in range(num_lesser):
            commands.extend(["Move " + w + " to the bottle." for w in less_distance_adjs])

        return commands

    elif feature_name == "distance_to_cube":
        traj1_feature_values = [distance_to_cube(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_cube(traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff/DISTANCE_TO_CUBE_STD))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            commands.extend(["Move " + w + " from the cube." for w in greater_distance_adjs])
        for i in range(num_lesser):
            commands.extend(["Move " + w + " to the cube." for w in less_distance_adjs])

        return commands
