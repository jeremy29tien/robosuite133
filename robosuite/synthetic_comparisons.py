import numpy as np
from robosuite.environments.manipulation.lift_features import speed, height, distance_to_bottle, distance_to_cube


greater_gtreward_adjs = ["better", "more successfully"]
less_gtreward_adjs = ["worse", "not as well"]
greater_speed_adjs = ["faster", "quicker", "swifter", "at a higher speed"]
less_speed_adjs = ["slower", "more moderate", "more sluggish", "at a lower speed"]
greater_height_adjs = ["higher", "taller", "at a greater height"]
less_height_adjs = ["lower", "shorter", "at a lesser height"]
greater_distance_adjs = ["further", "farther", "more distant"]
less_distance_adjs = ["closer", "nearer", "more nearby"]


# This function takes in two trajectories in the form of LISTS of (observation, action) pairs.
# Features:
# GT reward (mean)
# Speed (mean)
# Height (mean)
# Distance to cube (final)
# Distance to bottle (min)
def generate_synthetic_comparisons_statements(traj1, traj2, traj1_gtrewards, traj2_gtrewards, feature_name):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "gt_reward":
        traj1_feature_values = [traj1_gtrewards[t] for t in range(horizon)]
        traj2_feature_values = [traj2_gtrewards[t] for t in range(horizon)]

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
def generate_synthetic_comparisons_commands(traj1, traj2, traj1_gtrewards, traj2_gtrewards, feature_name):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "gt_reward":
        traj1_feature_values = [traj1_gtrewards[t] for t in range(horizon)]
        traj2_feature_values = [traj2_gtrewards[t] for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN gt_reward
            commands = ["Lift the cube " + w + "." for w in greater_gtreward_adjs]
            return commands
        else:
            commands = ["Lift the cube " + w + "." for w in less_gtreward_adjs]
            return commands

    elif feature_name == "speed":
        traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN speed
            commands = ["Move " + w + "." for w in greater_speed_adjs]
            return commands
        else:
            commands = ["Move " + w + "." for w in less_speed_adjs]
            return commands

    elif feature_name == "height":
        traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [height(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN height
            commands = ["Move " + w + "." for w in greater_height_adjs]
            return commands
        else:
            commands = ["Move " + w + "." for w in less_height_adjs]
            return commands

    elif feature_name == "distance_to_bottle":
        traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]

        # TODO: Later, we can make this non-Markovian (e.g., the MINIMUM distance)
        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN distance
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
        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):  # Here, we take the MEAN distance
            commands = ["Move " + w + " from the cube." for w in greater_distance_adjs]
            return commands
        else:
            commands = ["Move " + w + " to the cube." for w in less_distance_adjs]
            return commands

