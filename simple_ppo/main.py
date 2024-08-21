import os
import random
import time
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from func_to_script import script

from agent import ContinuousAgent, DiscreteAgent
from ppo import PPO, PPOLogger


def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def make_discrete_env(env_id, idx, capture_video, run_name):
    def create_configured_env():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return create_configured_env


def make_continuous_env(env_id, idx, capture_video, run_name, gamma):
    def create_configured_env():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return create_configured_env


def get_NormalizeObservation_wrapper(self, env_num=0):
    return self.gym_sync_vec_env.envs[env_num].env.env.env


def get_obs_norm_rms_obj(self, env_num=0):
    return self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms


def set_obs_norm_rms_obj(self, rms_obj, env_num=0):
    self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms = rms_obj


def get_obs_norm_rms_vars(self, env_num=0):
    rms_obj = self.get_obs_norm_rms_obj(env_num)
    return rms_obj.mean, rms_obj.var, rms_obj.count


def create_envs(env_id, num_envs, env_is_discrete, capture_video, run_name, gamma):
    if env_is_discrete:
        envs = gym.vector.SyncVectorEnv(
            [
                make_discrete_env(env_id, i, capture_video, run_name)
                for i in range(num_envs)
            ],
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"
    else:
        envs = gym.vector.SyncVectorEnv(
            [
                make_continuous_env(env_id, i, capture_video, run_name, gamma)
                for i in range(num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

    return envs


def load_and_evaluate_model(
    run_name,
    env_id,
    env_is_discrete,
    envs,
    num_envs,
    agent_class,
    device,
    model_path,
    gamma,
    capture_video,
):
    # Run simple evaluation to demonstrate how to load and use a trained model
    eval_episodes = 10
    eval_envs = create_envs(
        env_id=env_id,
        num_envs=1,
        env_is_discrete=env_is_discrete,
        capture_video=capture_video,
        run_name=f"{run_name}-eval",
        gamma=gamma,
    )

    if not env_is_discrete:
        # Update normalization stats for continuous environments
        avg_rms_obj = (
            np.mean([envs.get_obs_norm_rms_obj(i) for i in range(num_envs)]) / num_envs
        )
        eval_envs.set_obs_norm_rms_obj(avg_rms_obj)

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()

    obs, _ = eval_envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _ = eval_agent.sample_action_and_compute_log_prob(
            torch.Tensor(obs).to(device)
        )
        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"Eval episode {len(episodic_returns)}, episodic return: {info['episode']['r']}"
                    )
                    episodic_returns.append(info["episode"]["r"])

    eval_envs.close()


@script
def run_ppo(
    env_id: str = "CartPole-v1",
    env_is_discrete: bool = True,
    num_envs: int = 4,
    total_timesteps: int = 500000,
    num_rollout_steps: int = 128,
    update_epochs: int = 4,
    num_minibatches: int = 4,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    surrogate_clip_threshold: float = 0.2,
    entropy_loss_coefficient: float = 0.01,
    value_function_loss_coefficient: float = 0.5,
    normalize_advantages: bool = True,
    clip_value_function_loss: bool = True,
    max_grad_norm: float = 0.5,
    target_kl: float = None,
    anneal_lr: bool = True,
    rpo_alpha: float = None,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = False,
    save_model: bool = False,
):
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.

    This function sets up the environment, creates the PPO agent, and runs the training process.
    It handles both discrete and continuous action spaces, and includes options for
    various PPO algorithm parameters and training configurations.

    Args:
        # Environment parameters
        env_id (str): Identifier for the Gymnasium environment to use.
        env_is_discrete (bool): Whether the environment has a discrete action space.
        num_envs (int): Number of parallel environments to run.

        # Core training parameters
        total_timesteps (int): Total number of timesteps to run the training for. This is the number of interactions with the environment
        num_rollout_steps (int): Number of steps to run in each environment per rollout.
        update_epochs (int): Number of epochs to update the policy for each rollout.
        num_minibatches (int): Number of minibatches for each update.

        # Core PPO algorithm parameters
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.

        # Loss function coefficients
        entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
        value_function_loss_coefficient (float): Coefficient for the value function loss.

        # Advanced PPO parameters
        normalize_advantages (bool): Whether to normalize advantages.
        clip_value_function_loss (bool): Whether to clip the value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.
        target_kl (float): Target KL divergence for early stopping, if not None.

        # Learning rate schedule
        anneal_lr (bool): Whether to use learning rate annealing.

        # Continuous action space specific
        rpo_alpha (float): Alpha parameter for Regularized Policy Optimization (continuous action spaces only).

        # Reproducibility and logging
        seed (int): Random seed for reproducibility.
        torch_deterministic (bool): Whether to use deterministic algorithms in PyTorch.
        capture_video (bool): Whether to capture and save videos of the agent's performance.
        use_tensorboard (bool): Whether to use TensorBoard for logging.
        save_model (bool): Whether to save the trained model to disk and validate this by running a simple evaluation.
    """

    if env_is_discrete and rpo_alpha is not None:
        print(
            f"rpo_alpha is not used in discrete environments. Ignoring rpo_alpha={rpo_alpha}"
        )

    # Set up run name and logging
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

    set_seed(seed, torch_deterministic)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environments
    envs = create_envs(
        env_id, num_envs, env_is_discrete, capture_video, run_name, gamma
    )

    # Set up agent
    agent_class = (
        DiscreteAgent
        if env_is_discrete
        else partial(ContinuousAgent, rpo_alpha=rpo_alpha)
    )
    agent = agent_class(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        learning_rate=learning_rate,
        num_rollout_steps=num_rollout_steps,
        num_envs=num_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        surrogate_clip_threshold=surrogate_clip_threshold,
        entropy_loss_coefficient=entropy_loss_coefficient,
        value_function_loss_coefficient=value_function_loss_coefficient,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        normalize_advantages=normalize_advantages,
        clip_value_function_loss=clip_value_function_loss,
        target_kl=target_kl,
        anneal_lr=anneal_lr,
        envs=envs,
        seed=seed,
        logger=PPOLogger(run_name, use_tensorboard),
    )

    # Train the agent
    trained_agent = ppo.learn(total_timesteps)

    if save_model:
        model_path = f"runs/{run_name}/{exp_name}.rl_model"
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        load_and_evaluate_model(
            run_name,
            env_id,
            env_is_discrete,
            envs,
            num_envs,
            agent_class,
            device,
            model_path,
            gamma,
            capture_video,
        )

    # Close environments
    envs.close()


if __name__ == "__main__":
    run_ppo()
