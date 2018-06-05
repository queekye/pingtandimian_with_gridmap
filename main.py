from new_ddpg import DDPG
from env import Env
import time
import numpy as np

MAP_DIM = 64
PIXEL_METER = 32

env = Env(MAP_DIM, PIXEL_METER)
agent = DDPG(env.a_dim, env.s_dim, env.a_bound, MAP_DIM, PIXEL_METER, 2)

t1 = time.time()
replay_num = 0
zheng_reward = 0.
p = 0.1
success = np.zeros(100000)
for i in range(100000):
    t_start = time.time()
    sd = i * 3 + 100
    env.map_reset(p, i % 10000)
    s, loc = env.state_reset(sd)
    agent.exploration_noise.reset()
    ep_reward = 0
    ave_w = 0
    j = 0
    r = 0
    for j in range(200):
        # Add exploration noise
        a = agent.noise_action(s, env.map.map_matrix, loc)
        a_store = a.copy()
        a[:4] /= max(np.linalg.norm(a[:4]), 1e-8)
        a[-3:] *= 2
        ave_w += np.linalg.norm(a[-3:])
        a = np.minimum(2, np.maximum(-2, a))

        s_, loc_, r, done = env.step(a)
        agent.perceive(sd, p, loc, s, a_store, r, s_, loc_, done)
        replay_num += 1
        if r > 0:
            zheng_reward += 1
        s = s_
        loc = loc_
        ep_reward += r

        if done:
            if r == 10:
                success[i] = 1
            if i % 10000 == 0 and sum(success[max(i-10000, 0):i])/10000 > 0.7:
                p += 0.05
            break
    ave_w /= j+1
    print("episode: %6d   ep_reward:%8.5f   last_reward:%6.5f   replay_num:%8d   cost_time:%4.2f    ave_w:%8.5f    "
          "success_rate:%4f    正reward比例：%4f" % (i, ep_reward, r, replay_num, time.time() - t_start, ave_w,
                                                 sum(success) / (i + 1), zheng_reward / replay_num))
print('Running time: ', time.time() - t1)
