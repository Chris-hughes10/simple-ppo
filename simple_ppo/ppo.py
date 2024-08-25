from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class LinearLRSchedule:
    def __init__(self, optimizer, initial_lr, total_updates):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_updates = total_updates
        self.current_update = 0

    def step(self):
        self.current_update += 1
        frac = 1.0 - (self.current_update - 1.0) / self.total_updates
        lr = frac * self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False):
        self.use_tensorboard = use_tensorboard
        self.global_steps = []
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")

    def log_rollout_step(self, infos, global_step):
        self.global_steps.append(global_step)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}",
                        flush=True,
                    )

                    if self.use_tensorboard:
                        self.writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

    def log_policy_update(self, update_results, global_step):
        if self.use_tensorboard:
            self.writer.add_scalar(
                "losses/policy_loss", update_results["policy_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/value_loss", update_results["value_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/entropy_loss", update_results["entropy_loss"], global_step
            )

            self.writer.add_scalar(
                "losses/kl_divergence", update_results["old_approx_kl"], global_step
            )
            self.writer.add_scalar(
                "losses/kl_divergence", update_results["approx_kl"], global_step
            )
            self.writer.add_scalar(
                "losses/clipping_fraction",
                update_results["clipping_fractions"],
                global_step,
            )
            self.writer.add_scalar(
                "losses/explained_variance",
                update_results["explained_variance"],
                global_step,
            )


class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        learning_rate=3e-4,
        num_rollout_steps=2048,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.01,
        value_function_loss_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=32,
        normalize_advantages=True,
        clip_value_function_loss=True,
        target_kl=None,
        anneal_lr=True,
        seed=1,
        logger=None,
    ):
        """
        Proximal Policy Optimization (PPO) algorithm implementation.

        This class implements the PPO algorithm, a policy gradient method for reinforcement learning.
        It's designed to be more stable and sample efficient compared to traditional policy gradient methods.

        Args:
            agent: The agent (policy/value network) to be trained.
            optimizer: The optimizer for updating the agent's parameters.
            envs: The vectorized environment(s) to train on.

            # Core PPO parameters
            learning_rate (float): Learning rate for the optimizer.
            num_rollout_steps (int): Number of steps to run for each environment per update.
            num_envs (int): Number of parallel environments.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.
            entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
            value_function_loss_coefficient (float): Coefficient for the value function loss.
            max_grad_norm (float): Maximum norm for gradient clipping.

            # Training process parameters
            update_epochs (int): Number of epochs to update the policy for each rollout.
            num_minibatches (int): Number of minibatches to use per update.
            normalize_advantages (bool): Whether to normalize advantages.
            clip_value_function_loss (bool): Whether to clip the value function loss.
            target_kl (float or None): Target KL divergence for early stopping.

            # Learning rate schedule
            anneal_lr (bool): Whether to use learning rate annealing.

            # Reproducibility and tracking
            seed (int): Random seed for reproducibility of environment initialisation.
            logger (PPOLogger): A logger instance for logging. if None is passed, a default logger is created.

        The PPO algorithm works by collecting a batch of data from the environment,
        then performing multiple epochs of optimization on this data. It uses a surrogate
        objective function and a value function, both clipped to prevent too large policy updates.

        KL Divergence Approach:
        This implementation uses a fixed KL divergence threshold (`target_kl`) for early stopping.
        If `target_kl` is set (not None), the policy update will stop early if the approximate
        KL divergence exceeds this threshold. This acts as a safeguard against too large policy
        updates, helping to maintain the trust region.

        - If `target_kl` is None, no early stopping based on KL divergence is performed.
        - A smaller `target_kl` (e.g., 0.01) results in more conservative updates, potentially
          leading to more stable but slower learning.
        - A larger `target_kl` (e.g., 0.05) allows for larger policy updates, potentially
          leading to faster but possibly less stable learning.
        - Common values for `target_kl` range between 0.01 and 0.05. The original paper (https://arxiv.org/abs/1707.06347)
          settled on a target range of (0.003 to 0.03)

        The optimal `target_kl` can depend on the specific environment and problem. It's often
        beneficial to monitor the KL divergence during training and adjust `target_kl` based on
        the stability and speed of learning.
        """
        self.agent = agent
        self.envs = envs
        self.optimizer = optimizer
        self.seed = seed

        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.batch_size = num_envs * num_rollout_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_clip_threshold = surrogate_clip_threshold
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_function_loss_coefficient = value_function_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.normalize_advantages = normalize_advantages
        self.clip_value_function_loss = clip_value_function_loss
        self.target_kl = target_kl

        self.device = next(agent.parameters()).device

        self.anneal_lr = anneal_lr
        self.initial_lr = learning_rate

        self.lr_scheduler = None
        self._global_step = 0
        self.logger = logger or PPOLogger()

    def create_lr_scheduler(self, num_policy_updates):
        return LinearLRSchedule(self.optimizer, self.initial_lr, num_policy_updates)

    def learn(self, total_timesteps):
        """
        Train the agent using the PPO algorithm.

        This method runs the full training loop for the specified number of timesteps,
        collecting rollouts from the environment and updating the policy accordingly.

        Args:
            total_timesteps (int): The total number of environment timesteps to train for.

        Returns:
            agent: The trained agent (policy/value network).

        Process:
            1. Initialize the learning rate scheduler if annealing is enabled.
            2. Reset the environment to get initial observations.
            3. For each update iteration:
                a. Collect rollouts from the environment.
                b. Compute advantages and returns.
                c. Update the policy and value function multiple times on the collected data.
                d. Log relevant metrics and training progress.
            4. Return the trained agent.

        Notes:
            - The actual number of environment steps may be slightly less than `total_timesteps`
              due to the integer division when calculating the number of updates.
            - Early stopping based on KL divergence may occur if `target_kl` is set.
        """
        num_policy_updates = total_timesteps // (self.num_rollout_steps * self.num_envs)

        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_policy_updates)

        next_observation, is_next_observation_terminal = self._initialize_environment()

        # Initialize logging variables
        self._global_step = 0

        for update in range(num_policy_updates):
            if self.anneal_lr:
                self.lr_scheduler.step()

            (
                batch_observations,
                batch_log_probabilities,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_values,
                next_observation,
                is_next_observation_terminal,
            ) = self.collect_rollouts(next_observation, is_next_observation_terminal)

            update_results = self.update_policy(
                batch_observations,
                batch_log_probabilities,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_values,
            )

            self.logger.log_policy_update(update_results, self._global_step)

        print(f"Training completed. Total steps: {self._global_step}")

        return self.agent  # Return the trained agent

    def _initialize_environment(self):
        """
        Initialize the environment for the start of training, resets the vectorized environments
        to their initial states and prepares the initial observation and termination flag for the agent to begin
        interacting with the environment.

        Returns:
            tuple: A tuple containing:
                - initial_observation (torch.Tensor): The initial observation from the
                environment
                - is_initial_observation_terminal (torch.Tensor): A tensor of zeros
                indicating that the initial state is not terminal.

        Note:
            The method uses the seed set during the PPO initialization to ensure
            reproducibility of the environment's initial state across different runs.
        """
        initial_observation, _ = self.envs.reset(seed=self.seed)
        initial_observation = torch.Tensor(initial_observation).to(self.device)
        is_initial_observation_terminal = torch.zeros(self.num_envs).to(self.device)
        return initial_observation, is_initial_observation_terminal

    def collect_rollouts(self, next_observation, is_next_observation_terminal):
        """
        Collect a set of rollout data by interacting with the environment. A rollout is a sequence of observations,
        actions, and rewards obtained by running the current policy in the environment. The collected data is crucial
        for the subsequent policy update step in the PPO algorithm.

        This method performs multiple timesteps across all (parallel) environments, collecting data which
        will be used to update the policy.

        The collected data represents a fixed number of timesteps (num_rollout_steps * num_envs)
        of interaction, which forms the basis for a single PPO update iteration. This may
        include partial trajectories from multiple episodes across the parallel environments.

        The method uses the current policy to select actions, executes these actions in the
        environment, and stores the resulting observations, rewards, and other relevant
        information. It also computes advantages and returns using Generalized Advantage
        Estimation (GAE), which are crucial for the PPO algorithm.

        Args:
            next_observation (torch.Tensor): The starting observation for this rollout.
            is_next_observation_terminal (torch.Tensor): Boolean tensor indicating whether
                the starting state is terminal.

        Returns:
            tuple: A tuple containing:
                - batch_observations (torch.Tensor): Flattened tensor of all observations.
                - batch_log_probabilities (torch.Tensor): Log probabilities of the actions taken.
                - batch_actions (torch.Tensor): Flattened tensor of all actions taken.
                - batch_advantages (torch.Tensor): Computed advantage estimates.
                - batch_returns (torch.Tensor): Computed returns (sum of discounted rewards).
                - batch_values (torch.Tensor): Value function estimates for each state.
                - next_observation (torch.Tensor): The final observation after collecting rollouts.
                - is_next_observation_terminal (torch.Tensor): Whether the final state is terminal.

        Note:
            This method updates the global step during the rollout collection process.
        """
        (
            collected_observations,
            actions,
            action_log_probabilities,
            rewards,
            is_episode_terminated,
            observation_values,
        ) = self._initialize_storage()

        for step in range(self.num_rollout_steps):
            # Store current observation
            collected_observations[step] = next_observation
            is_episode_terminated[step] = is_next_observation_terminal

            with torch.no_grad():
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    next_observation
                )
                value = self.agent.estimate_value_from_observation(next_observation)

                observation_values[step] = value.flatten()
            actions[step] = action
            action_log_probabilities[step] = logprob

            # Execute the environment and store the data
            next_observation, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            self._global_step += self.num_envs
            rewards[step] = torch.as_tensor(reward, device=self.device).view(-1)
            is_next_observation_terminal = np.logical_or(terminations, truncations)

            next_observation, is_next_observation_terminal = (
                torch.as_tensor(
                    next_observation, dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    is_next_observation_terminal,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

            self.logger.log_rollout_step(infos, self._global_step)

        # Estimate the value of the next state (the state after the last collected step) using the current policy
        # This value will be used in the GAE calculation to compute advantages
        with torch.no_grad():
            next_value = self.agent.estimate_value_from_observation(
                next_observation
            ).reshape(1, -1)

            advantages, returns = self.compute_advantages(
                rewards,
                observation_values,
                is_episode_terminated,
                next_value,
                is_next_observation_terminal,
            )

        # Flatten the batch for easier processing in the update step
        (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
        ) = self._flatten_rollout_data(
            collected_observations,
            action_log_probabilities,
            actions,
            advantages,
            returns,
            observation_values,
        )

        # Return the collected and computed data for the policy update step
        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
            next_observation,
            is_next_observation_terminal,
        )

    def _initialize_storage(self):
        collected_observations = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
            + self.envs.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
            + self.envs.single_action_space.shape
        ).to(self.device)
        action_log_probabilities = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
        ).to(self.device)
        rewards = torch.zeros((self.num_rollout_steps, self.num_envs)).to(self.device)
        is_episode_terminated = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        observation_values = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )

        return (
            collected_observations,
            actions,
            action_log_probabilities,
            rewards,
            is_episode_terminated,
            observation_values,
        )

    def compute_advantages(
        self,
        rewards,
        values,
        is_observation_terminal,
        next_value,
        is_next_observation_terminal,
    ):
        """
        Compute advantages using Generalized Advantage Estimation (GAE). The advantage function
        measures how much better an action is compared to the average action for a given observation.
        This helps reduce variance in gradient estimates while maintaining a tolerable level of bias
        for policy gradient methods.

        This method implements GAE, which provides a good trade-off between bias and
        variance in advantage estimation. GAE uses a weighted average of n-step
        returns to compute the advantage, controlled by the lambda parameter.

        The GAE is computed as: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

        Args:
            rewards (torch.Tensor): Tensor of shape (num_steps, num_envs) containing
                the rewards for each step in the rollout.
            values (torch.Tensor): Tensor of shape (num_steps, num_envs) containing
                the value estimates for each state in the rollout.
            is_observation_terminal (torch.Tensor): Tensor of shape (num_steps, num_envs)
                indicating whether each state is terminal.
            next_value (torch.Tensor): The estimated value of the state following the
                last state in the rollout.
            is_next_observation_terminal (torch.Tensor): Whether the next state
                (after the last in the rollout) is terminal.

        Returns:
            tuple: A tuple containing:
                - advantages (torch.Tensor): Tensor of shape (num_steps, num_envs)
                containing the computed advantage estimates.
                - returns (torch.Tensor): Tensor of shape (num_steps, num_envs)
                containing the computed returns (sum of discounted rewards).

        This implementation computes GAE iteratively in reverse order to efficiently
        handle multi-step advantages.

        """
        # Initialize advantage estimates
        advantages = torch.zeros_like(rewards).to(self.device)

        # Initialize the GAE accumulator
        gae_running_value = 0

        # Accumulate the GAE values in reverse order, which allows us to efficiently
        # compute multi-step advantages
        for t in reversed(range(self.num_rollout_steps)):
            if t == self.num_rollout_steps - 1:
                # For the last step, use the provided next_value and whether it is terminal
                episode_continues = 1.0 - is_next_observation_terminal
                next_values = next_value
            else:
                # For all other steps, use values and terminal flags from the next step
                episode_continues = 1.0 - is_observation_terminal[t + 1]
                next_values = values[t + 1]

            # Compute TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - is_terminal_{t+1}) - V(s_t)
            # The TD error represents the difference between
            # the expected value (current reward plus discounted next observation value)
            # and the estimated value of the current observation. It's a measure of
            # how "surprised" we are by the outcome of an action.
            delta = (
                rewards[t] + self.gamma * next_values * episode_continues - values[t]
            )

            # Compute GAE:  A_t = δ_t + (γλ) * (1 - is_terminal_{t+1}) * A_{t+1}
            # Intuition: GAE is a weighted sum of TD errors, with more recent
            # errors weighted more heavily. This balances between using only
            # the immediate reward (high variance, unbiased) and using the
            # full return (low variance, biased). The λ parameter controls
            # this trade-off.
            advantages[t] = gae_running_value = (
                delta
                + self.gamma * self.gae_lambda * episode_continues * gae_running_value
            )

        # The return is the sum of the advantage (how much better the action was than expected)
        # and the value estimate (what we expected).
        # Combining our value function estimate with the computed advantage gives
        # an estimate of the total return from each observation.
        # This works because:
        # 1. Advantage A(s,a) = Q(s,a) - V(s)
        # 2. Q(s,a) represents the expected return from taking action a in state s
        # 3. By adding V(s) back to A(s,a), we reconstruct an estimate of the full return
        returns = advantages + values

        return advantages, returns

    def _flatten_rollout_data(
        self,
        collected_observations,
        action_log_probabilities,
        actions,
        advantages,
        returns,
        observation_values,
    ):
        batch_observations = collected_observations.reshape(
            (-1,) + self.envs.single_observation_space.shape
        )
        batch_log_probabilities = action_log_probabilities.reshape(-1)
        batch_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = observation_values.reshape(-1)

        return (
            batch_observations,
            batch_log_probabilities,
            batch_actions,
            batch_advantages,
            batch_returns,
            batch_values,
        )

    def update_policy(
        self,
        collected_observations,
        collected_action_log_probs,
        collected_actions,
        computed_advantages,
        computed_returns,
        previous_value_estimates,
    ):
        """
        Update the policy and value function using the collected rollout data.

        This method implements the core PPO algorithm update step. It performs multiple
        epochs of updates on minibatches of the collected rollout data, optimizing both the policy
        and value function.

        Args:
            collected_observations (torch.Tensor): Tensor of shape (batch_size, *obs_shape)
                containing the observations from the rollout.
            collected_action_log_probs (torch.Tensor): Tensor of shape (batch_size,)
                containing the log probabilities of the actions taken during the rollout.
            collected_actions (torch.Tensor): Tensor of shape (batch_size, *action_shape)
                containing the actions taken during the rollout.
            computed_advantages (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed advantages for each step in the rollout.
            computed_returns (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed returns (sum of discounted rewards) for each step.
            previous_value_estimates (torch.Tensor): Tensor of shape (batch_size,)
                containing the value estimates from the previous iteration.

        Returns:
            dict: A dictionary containing various statistics about the update process:
                - policy_loss: The final policy gradient loss.
                - value_loss: The final value function loss.
                - entropy_loss: The entropy loss, encouraging exploration.
                - old_approx_kl: The old approximate KL divergence.
                - approx_kl: The new approximate KL divergence.
                - clipping_fraction: The fraction of policy updates that were clipped.
                - explained_variance: A measure of how well the value function explains
                                    the observed returns.

        The method performs the following key steps:
        1. Iterates over the data for multiple epochs, shuffling at each epoch.
        2. For each minibatch:
            a. Computes new action probabilities and values.
            b. Calculates the policy ratio and clipped policy objective.
            c. Computes the value function loss, optionally with clipping.
            d. Calculates the entropy bonus to encourage exploration.
            e. Combines losses and performs a gradient update step.
        3. Optionally performs early stopping based on KL divergence.
        4. Computes and returns various statistics about the update process.

        This implementation uses the PPO clipped objective, which helps to constrain
        the policy update and improve training stability. It also uses advantage
        normalization and gradient clipping for further stability.
        """
        # Prepare for minibatch updates
        batch_size = self.num_rollout_steps * self.num_envs
        batch_indices = np.arange(batch_size)

        # Track clipping for monitoring policy update magnitude
        clipping_fractions = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_indices)

            # Minibatch updates help stabilize training and can be more compute-efficient
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Get updated action probabilities and values for the current policy
                current_policy_log_probs, entropy = (
                    self.agent.compute_action_log_probabilities_and_entropy(
                        collected_observations[minibatch_indices],
                        collected_actions[minibatch_indices],
                    )
                )
                new_value = self.agent.estimate_value_from_observation(
                    collected_observations[minibatch_indices]
                )

                # Calculate the probability ratio for importance sampling
                # This allows us to use old trajectories to estimate the new policy's performance
                log_probability_ratio = (
                    current_policy_log_probs
                    - collected_action_log_probs[minibatch_indices]
                )
                probability_ratio = log_probability_ratio.exp()

                # Estimate KL divergence for early stopping
                # This helps prevent the new policy from diverging too much from the old policy
                # approx_kl http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    old_approx_kl = (-log_probability_ratio).mean()
                    approx_kl = ((probability_ratio - 1) - log_probability_ratio).mean()

                    # Track the fraction of updates being clipped
                    # High clipping fraction might indicate too large policy updates
                    clipping_fractions += [
                        (
                            (probability_ratio - 1.0).abs()
                            > self.surrogate_clip_threshold
                        )
                        .float()
                        .mean()
                        .item()
                    ]

                minibatch_advantages = computed_advantages[minibatch_indices]

                if self.normalize_advantages:
                    # Normalize advantages to reduce variance in updates
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                policy_gradient_loss = self.calculate_policy_gradient_loss(
                    minibatch_advantages, probability_ratio
                )
                value_function_loss = self.calculate_value_function_loss(
                    new_value,
                    computed_returns,
                    previous_value_estimates,
                    minibatch_indices,
                )
                # Entropy encourages exploration by penalizing overly deterministic policies
                entropy_loss = entropy.mean()

                # Combine losses: minimize policy and value losses, maximize entropy
                loss = (
                    policy_gradient_loss
                    - self.entropy_loss_coefficient
                    * entropy_loss  # subtraction here to maximise entropy (exploration)
                    + value_function_loss * self.value_function_loss_coefficient
                )

                # Perform backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping helps prevent too large policy updates
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping based on KL divergence, if enabled, done at epoch level for stability
            # This provides an additional safeguard against too large policy updates
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        predicted_values, actual_returns = (
            previous_value_estimates.cpu().numpy(),
            computed_returns.cpu().numpy(),
        )
        observed_return_variance = np.var(actual_returns)
        # explained variance measures how well the value function predicts the actual returns
        explained_variance = (
            np.nan
            if observed_return_variance == 0
            else 1
            - np.var(actual_returns - predicted_values) / observed_return_variance
        )

        return {
            "policy_loss": policy_gradient_loss.item(),
            "value_loss": value_function_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipping_fractions": np.mean(clipping_fractions),
            "explained_variance": explained_variance,
        }

    def calculate_policy_gradient_loss(self, minibatch_advantages, probability_ratio):
        """
        Calculate the policy gradient loss using the PPO clipped objective, which is designed to
        improve the stability of policy updates. It uses a clipped surrogate objective
        that limits the incentive for the new policy to deviate too far from the old policy.

        Args:
            minibatch_advantages (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the advantage estimates for each sample in the minibatch.
            probability_ratio (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the ratio of probabilities under the new and old policies for each action.

        Returns:
            torch.Tensor: A scalar tensor containing the computed policy gradient loss.

        The PPO loss is defined as:
        L^CLIP(θ) = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        Where:
        - r_t(θ) is the probability ratio
        - A_t is the advantage estimate
        - ε is the surrogate_clip_threshold
        """

        # L^PG(θ) = r_t(θ) * A_t
        # This is the standard policy gradient objective. It encourages
        # the policy to increase the probability of actions that led to higher
        # advantages (i.e., performed better than expected).
        unclipped_pg_obj = minibatch_advantages * probability_ratio

        # L^CLIP(θ) = clip(r_t(θ), 1-ε, 1+ε) * A_t
        # This limits how much the policy can change for each action.
        # If an action's probability increased/decreased too much compared to
        # the old policy, we clip it. This prevents drastic policy changes,
        # promoting more stable learning.
        clipped_pg_obj = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )

        # L^CLIP(θ) = -min(L^PG(θ), L^CLIP(θ))
        # Use the minimum of the clipped and unclipped objectives.
        # By taking the minimum and then negating (for gradient ascent),
        # we choose the more pessimistic (lower) estimate.
        # This ensures that:
        # 1. We don't overly reward actions just because they had high advantages
        #    (unclipped loss might do this).
        # 2. We don't ignore actions where the policy changed a lot if they still
        #    result in a worse objective (clipped loss might do this).
        # This conservative approach helps prevent the policy from changing too
        # rapidly in any direction, improving stability.
        policy_gradient_loss = -torch.min(unclipped_pg_obj, clipped_pg_obj).mean()

        return policy_gradient_loss

    def calculate_value_function_loss(
        self, new_value, computed_returns, previous_value_estimates, minibatch_indices
    ):
        """
        Calculate the value function loss, optionally with clipping, for the value function approximation.
        It uses either a simple MSE loss or a clipped version similar to the policy loss clipping
        in PPO. When clipping is enabled, it uses the maximum of clipped and unclipped losses.
        The clipping helps to prevent the value function from changing too much in a single update.


        Args:
            new_value (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the new value estimates for the sampled states.
            computed_returns (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed returns for each step in the rollout.
            previous_value_estimates (torch.Tensor): Tensor of shape (batch_size,)
                containing the value estimates from the previous iteration.
            minibatch_indices (np.array): Array of indices for the current minibatch.

        Returns:
            torch.Tensor: A scalar tensor containing the computed value function loss.

        The value function loss is defined as:
        If clipping is enabled:
        L^VF = 0.5 * E[max((V_θ(s_t) - R_t)^2, (clip(V_θ(s_t) - V_old(s_t), -ε, ε) + V_old(s_t) - R_t)^2)]
        If clipping is disabled:
        L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]

        Where:
        - V_θ(s_t) is the new value estimate
        - R_t is the computed return
        - V_old(s_t) is the old value estimate
        - ε is the surrogate_clip_threshold
        """
        new_value = new_value.view(-1)

        if self.clip_value_function_loss:
            # L^VF_unclipped = (V_θ(s_t) - R_t)^2
            # This is the standard MSE loss, pushing the value estimate
            # towards the actual observed returns.
            unclipped_vf_loss = (new_value - computed_returns[minibatch_indices]) ** 2

            # V_clipped = V_old(s_t) + clip(V_θ(s_t) - V_old(s_t), -ε, ε)
            # This limits how much the value estimate can change from its
            # previous value, promoting stability in learning.
            clipped_value_diff = torch.clamp(
                new_value - previous_value_estimates[minibatch_indices],
                -self.surrogate_clip_threshold,
                self.surrogate_clip_threshold,
            )
            clipped_value = (
                previous_value_estimates[minibatch_indices] + clipped_value_diff
            )

            # L^VF_clipped = (V_clipped - R_t)^2
            # This loss encourages updates within the clipped range, preventing drastic changes to the value function.
            clipped_vf_loss = (clipped_value - computed_returns[minibatch_indices]) ** 2

            # L^VF = max(L^VF_unclipped, L^VF_clipped)
            # By taking the maximum, we choose the more pessimistic (larger) loss.
            # This ensures we don't ignore large errors outside the clipped range
            # while still benefiting from clipping's stability.
            v_loss_max = torch.max(unclipped_vf_loss, clipped_vf_loss)

            # The 0.5 factor simplifies the gradient of the squared error loss,
            # as it cancels out with the 2 from the derivative of x^2.
            value_function_loss = 0.5 * v_loss_max.mean()
        else:
            # If not clipping, use simple MSE loss
            # L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]
            # Intuition: Without clipping, we directly encourage the value function
            # to predict the observed returns as accurately as possible.
            value_function_loss = (
                0.5 * ((new_value - computed_returns[minibatch_indices]) ** 2).mean()
            )

        return value_function_loss
