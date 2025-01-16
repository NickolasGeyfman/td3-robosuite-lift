import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(
        self, actor_learning_rate, critic_learning_rate, input_dims, tau, env,
        gamma=0.99, update_actor_interval=2, warmup=1000,
        n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128,
        batch_size=100, noise=0.1
    ):
        self.gamma = gamma
        self.tau = tau
        
        # Additional TD3 hyperparams for target noise
        self.policy_noise = 0.2     # noise added to target actions
        self.noise_clip = 0.5       # clamp target noise
        self.noise = noise          # exploration noise for actor

        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        # Default device to CPU; can be overridden after creation
        self.device = T.device('cpu')

        # Create networks
        self.actor = ActorNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='actor', learning_rate=actor_learning_rate
        ).to(self.device)

        self.critic_1 = CriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='critic_1', learning_rate=critic_learning_rate
        ).to(self.device)
        
        self.critic_2 = CriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='critic_2', learning_rate=critic_learning_rate
        ).to(self.device)

        self.target_actor = ActorNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='target_actor', learning_rate=actor_learning_rate
        ).to(self.device)
        
        self.target_critic_1 = CriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='target_critic_1', learning_rate=critic_learning_rate
        ).to(self.device)
        
        self.target_critic_2 = CriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='target_critic_2', learning_rate=critic_learning_rate
        ).to(self.device)
        
        # Initialize target networks to match the main networks
        self.update_network_parameters(tau=1.0)

    def choose_action(self, observation, validation=False):
        """
        Decide the next action. If we're still in warmup, sample random actions.
        Otherwise, use the actor + exploration noise.
        """
        if self.time_step < self.warmup and not validation:
            # Random exploration during warmup
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,)),
                dtype=T.float32, device=self.device
            )
        else:
            # Use the actor to generate an action
            state = T.tensor(observation, dtype=T.float).to(self.device)
            mu = self.actor.forward(state)
        
        # Add exploration noise (per-dimension) if we are training
        if not validation:
            exploration_noise = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,)),
                dtype=T.float32, device=self.device
            )
            mu_prime = mu + exploration_noise
        else:
            mu_prime = mu

        # Clamp action to valid range
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)

        # Target actions from target actor
        target_actions = self.target_actor.forward(next_state)

        # Add clipped noise to target actions (TD3 style)
        noise = T.tensor(
            np.random.normal(scale=self.policy_noise, size=target_actions.shape),
            dtype=T.float32, device=self.device
        )
        noise = T.clamp(noise, -self.noise_clip, self.noise_clip)
        target_actions = target_actions + noise
        # Clamp final target action
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Zero Q for terminal states
        next_q1[done] = 0.0
        next_q2[done] = 0.0

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        # Min of the two target critics
        next_critic_value = T.min(next_q1, next_q2)
        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        # Critic update
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # Delayed actor update
        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Actor
        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in actor_params:
            target_actor_params[name] = tau * actor_params[name].clone() + \
                (1 - tau) * target_actor_params[name].clone()
        self.target_actor.load_state_dict(target_actor_params)

        # Critic 1
        critic_1_params = dict(self.critic_1.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        for name in critic_1_params:
            target_critic_1_params[name] = tau * critic_1_params[name].clone() + \
                (1 - tau) * target_critic_1_params[name].clone()
        self.target_critic_1.load_state_dict(target_critic_1_params)

        # Critic 2
        critic_2_params = dict(self.critic_2.named_parameters())
        target_critic_2_params = dict(self.target_critic_2.named_parameters())
        for name in critic_2_params:
            target_critic_2_params[name] = tau * critic_2_params[name].clone() + \
                (1 - tau) * target_critic_2_params[name].clone()
        self.target_critic_2.load_state_dict(target_critic_2_params)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        print("Successfully saved models")

    def load_models(self):
        try:
            checkpoint = T.load(self.actor.checkpoint_file, map_location=self.device)
            self.actor.load_state_dict(checkpoint)
            self.actor.to(self.device)

            checkpoint = T.load(self.critic_1.checkpoint_file, map_location=self.device)
            self.critic_1.load_state_dict(checkpoint)
            self.critic_1.to(self.device)

            checkpoint = T.load(self.critic_2.checkpoint_file, map_location=self.device)
            self.critic_2.load_state_dict(checkpoint)
            self.critic_2.to(self.device)

            checkpoint = T.load(self.target_actor.checkpoint_file, map_location=self.device)
            self.target_actor.load_state_dict(checkpoint)
            self.target_actor.to(self.device)

            checkpoint = T.load(self.target_critic_1.checkpoint_file, map_location=self.device)
            self.target_critic_1.load_state_dict(checkpoint)
            self.target_critic_1.to(self.device)

            checkpoint = T.load(self.target_critic_2.checkpoint_file, map_location=self.device)
            self.target_critic_2.load_state_dict(checkpoint)
            self.target_critic_2.to(self.device)

            print("Successfully loaded models")

        except Exception as e:
            print(f"Failed to load models: {e}")