import numpy as np

"""
A collection of helper methods that implement features (to be used later down the line for comparisons)
for the **LiftModded** task.
"""


def gt_reward(gym_obs):
    assert len(gym_obs) == 64 or len(
        gym_obs) == 68  # Ensure that we are using the right observation (64) or observation+action (68) space.

    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    reward = 0.0

    # Check success (sparse completion reward)
    cube_pos = object_state[0:3]
    cube_height = cube_pos[2]
    if cube_height > 0.04:  # refer to line 428 in lift.py. here, we assume table_height == 0
        reward = 2.25

    else:
        # reaching reward
        dist = distance_to_cube(gym_obs)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        reward += reaching_reward

        # grasping reward

    return reward


def speed(gym_obs):
    assert len(gym_obs) == 64 or len(gym_obs) == 68  # Ensure that we are using the right observation (64) or observation+action (68) space.
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    # joint_vel = proprio_state[14:21]
    # gripper_qvel = proprio_state[34:40]
    # efc_vel = object_state[20]
    hand_vel = object_state[21:24]

    # return np.linalg.norm(gripper_qvel, 2)  # Returning L2 norm of the gripper q-velocities as the speed.
    # return efc_vel
    return np.linalg.norm(hand_vel, 2)  # Returning L2 norm of the hand velocities as the speed.


prev_eef_pos = np.zeros(3)
def finite_diff_speed(gym_obs):
    global prev_eef_pos
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]
    eef_pos = proprio_state[21:24]
    vel = eef_pos - prev_eef_pos
    prev_eef_pos = eef_pos
    return np.linalg.norm(vel, 2)


def height(gym_obs):
    assert len(gym_obs) == 64 or len(gym_obs) == 68  # Ensure that we are using the right observation (64) or observation+action (68) space.
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    eef_pos = proprio_state[21:24]
    return eef_pos[2]  # Returning z-component of end-effector position as height.


def distance_to_bottle(gym_obs):
    assert len(gym_obs) == 64 or len(gym_obs) == 68  # Ensure that we are using the right observation (64) or observation+action (68) space.
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    gripper_to_bottle_pos = object_state[17:20]
    return np.linalg.norm(gripper_to_bottle_pos, 2)


def distance_to_cube(gym_obs):
    assert len(gym_obs) == 64 or len(gym_obs) == 68  # Ensure that we are using the right observation (64) or observation+action (68) space.
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    gripper_to_cube_pos = object_state[7:10]
    return np.linalg.norm(gripper_to_cube_pos, 2)
