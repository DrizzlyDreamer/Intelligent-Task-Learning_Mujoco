# import mujoco_py
# import os
# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)
#
# print(sim.data.qpos)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#
# sim.step()
# print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]

import gym
import mujoco_py
# os.environ.get("LD_LIBRARY_PATH", "")
# env = gym.make("LunarLander-v2", render_mode="human")

env = gym.make('Ant-v2', ctrl_cost_weight=0.1,render_mode="human")
# env = gym.make('InvertedDoublePendulum-v4',render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(10000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(observation)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

