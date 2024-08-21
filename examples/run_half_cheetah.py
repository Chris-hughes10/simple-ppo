
from main import run_ppo

if __name__ == '__main__':
     run_ppo(
        env_id="HalfCheetah-v4",
        total_timesteps=1000000,
        learning_rate=3e-4,
        num_envs=1,
        num_rollout_steps=2048,
        num_minibatches=32,
        update_epochs=10,
        entropy_loss_coefficient=0.0,
        env_is_discrete=False,
        capture_video=True,
        use_tensorboard=True,
    ) 