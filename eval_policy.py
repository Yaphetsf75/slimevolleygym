import gym
import torch
import slimevolleygym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(policy1,policy2, env='SlimeVolley-v0', num_test_episodes=10, render=False, verbose=False):
    test_env = gym.make(env)
    test_rewards1 = []
    test_rewards2 = []
    for i in range(num_test_episodes):
        state = test_env.reset()
        episode_total_reward1 = 0
        episode_total_reward2 = 0
        while True:
            state = torch.tensor([state], device=device, dtype=torch.float32)
            action1 = policy1.select_action(state).cpu().numpy()[0][0]
            action2 = policy2.select_action(state).cpu().numpy()[0][0]
            next_state, reward, done, _ = test_env.step(action1,action2)
            
            if render:
                test_env.render(mode='human')
            
            episode_total_reward1 += reward
            episode_total_reward2 -= reward
            state = next_state
            if done:
                if verbose:
                    print('[Episode {:4d}/{}] [reward {:.1f}]'
                        .format(i, num_test_episodes, episode_total_reward))
                break
        test_rewards1.append(episode_total_reward1)
        test_rewards2.append(episode_total_reward2)
    test_env.close()
    return sum(test_rewards1),sum(test_rewards2)


if __name__ == "__main__":
    import argparse
    from model import MyModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None, type=str,
        help='Path to the model weights.')
    parser.add_argument('--env', default=None, type=str,
        help='Name of the environment.')
    
    args = parser.parse_args()
    env = gym.make(args.env)
    model1 = MyModel(state_size=len(env.reset()), action_size=8)
    model1.load_state_dict(torch.load(args.model_path))
    model1 = model.to(device)
    model2 = MyModel(state_size=len(env.reset()), action_size=8)
    model2.load_state_dict(torch.load(args.model_path))
    model2 = model.to(device)
    env.close()

    eval_policy(policy1=model1,policy2=model2, env=args.env, render=True, verbose=True)
