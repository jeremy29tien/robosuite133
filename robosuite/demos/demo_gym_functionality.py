"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI 
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.lift_features import speed, finite_diff_speed, height, distance_to_bottle, distance_to_cube
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "LiftModded",
            robots="Jaco",
            use_object_obs=True,
            use_camera_obs=True,  # do not use pixel observations
            has_offscreen_renderer=True,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            controller='OSC_POSITION',
        )
    )

    # for i_episode in range(1):
    observation = env.reset()
    speeds = []
    fd_speeds = []
    for t in range(500):
        #env.render()
        action = env.action_space.sample()
        print("obs:", len(observation))
        print("action:", len(action))
        observation, reward, done, info = env.step(action)
        speeds.append(speed(observation))
        fd_speeds.append(finite_diff_speed(observation))
        print("speed:", speeds[-1])
        print("speed via finite differences:", fd_speeds[-1])
        print("height:", height(observation))
        print("distance to bottle:", distance_to_bottle(observation))
        print("distance to cube:", distance_to_cube(observation))
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    # fig1 = plt.figure("speeds")
    # plt.plot(speeds)
    #
    # fd_speeds.pop(0)  # The first finite diff value won't make sense, since our initial prev_eef_pos is np.zeros(3).
    # fig2 = plt.figure("fd_speeds")
    # plt.plot(fd_speeds)
    # plt.show()
