import numpy as np

"""
A collection of helper methods that implement features (to be used later down the line for comparisons)
for the **LiftModded** task.
"""


def speed(gym_obs):
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
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    eef_pos = proprio_state[21:24]
    return eef_pos[2]  # Returning z-component of end-effector position as height.


def distance_to_bottle(gym_obs):
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    gripper_to_bottle_pos = object_state[17:20]
    return np.linalg.norm(gripper_to_bottle_pos, 2)


def distance_to_cube(gym_obs):
    object_state = gym_obs[0:24]
    proprio_state = gym_obs[24:64]

    gripper_to_cube_pos = object_state[7:10]
    return np.linalg.norm(gripper_to_cube_pos, 2)
