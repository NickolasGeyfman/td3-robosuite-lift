import time
import os 
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork  # Fixed typo in import
from buffer import ReplayBuffer
from td3_torch import Agent
import torch as T

if __name__ == "__main__":

    # Create directory for saving models if it doesn't exist
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    # Create the environment using robosuite
    env = suite.make(
        env_name, 
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap the environment to make it compatible with Gym
    env = GymWrapper(env)

    # These parameters are derived from experimentation or from the paper
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    # Creating an Agent
    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size)
    agent.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')

    # Initializes a SummaryWriter object from the torch.utils.tensorboard library. It's used to log the training process
    writer = SummaryWriter("logs")
    # Sets the number of games or episodes to be played
    n_games = 10000
    # Will be used to keep track of the best score achieved by the agent
    best_score = 0
    # Creates a unique identifier for the episode
    episode_identifier = f"actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} batch_size={batch_size} layer1_size={layer1_size} layer2_size={layer2_size} tau=0.005 env_name={env_name} {int(time.time())}"

    # Loads pre-trained models or previously saved models from the disk
    agent.load_models()

    # This line starts a loop that will run for n_games iterations. Each iteration represents a single game or episode in the environment.
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        # Run the episode until done
        while not done:
            action = agent.choose_action(observation)  # Fixed typo: observations -> observation
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation  # Update the observation

        # Log the score for the episo
        
        writer.add_scalar(f"score/{episode_identifier}", score, global_step=i)

        # Save models every 10 episodes
        if i % 10 == 0:  # Fixed condition to save every 10 episodes
            agent.save_models()
        
        print(f"Episode {i}, Score {score}")