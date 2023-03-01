import hydra
import numpy as np
from tqdm import tqdm

import vima_bench

import matplotlib.pyplot as plt


@hydra.main(config_path=".", config_name="conf")
def main(cfg):
    kwargs = cfg.vima_bench_kwargs
    seed = kwargs["seed"]

    env = vima_bench.make(**kwargs)
    task = env.task
    oracle_fn = task.oracle(env)

    for _ in tqdm(range(1)):
        env.seed(seed)

        obs = env.reset()
        env.render()
        prompt, prompt_assets = env.get_prompt_and_assets()
        print("Prompt: ", prompt)
        print("Prompt assets: ", prompt_assets)
        print("Goal: ", task.goals)
        plot_obs(obs)
        
        for _ in range(task.oracle_max_steps):
            oracle_action = oracle_fn.act(obs)
            # clamp action to valid range
            oracle_action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in oracle_action.items()
            }
            obs, reward, done, info = env.step(action=oracle_action, skip_oracle=False)
            print(reward, done, info)
            plot_obs(obs)
            if done:
                break
        seed += 1

def plot_obs(obs):
    _, axs = plt.subplots(2,2, figsize=(8,10))
    axs[0,0].imshow(obs['rgb']['top'].transpose(1, 2, 0))
    axs[0,0].axis('off')
    axs[0,1].imshow(obs['rgb']['front'].transpose(1, 2, 0))
    axs[0,1].axis('off')
    
    axs[1,0].imshow(obs['segm']['top'])
    axs[1,0].axis('off')
    axs[1,1].imshow(obs['segm']['front'])
    axs[1,1].axis('off')
    plt.show()

if __name__ == "__main__":
    main()
