import time
import os 
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
import torch as T

if __name__ == "__main__":

    # Create directory for saving models if it doesn't exist
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    # Switch to Lift environment
    env_name = "Lift"

    # Create the environment using robosuite, with a smaller horizon for faster training
    env = suite.make(
        env_name, 
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,         # <-- Keep headless for faster training
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=100,               # <-- Reduced horizon for quicker rollouts
    )

    # Wrap the environment to make it compatible with Gym
    env = GymWrapper(env)

    # Hyperparameters for TD3
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    # Create an Agent
    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size
    )
    
    # Device selection: prefer CUDA -> MPS -> CPU
    if T.cuda.is_available():
        agent.device = T.device("cuda")
    elif T.backends.mps.is_available():
        agent.device = T.device("mps")
    else:
        agent.device = T.device("cpu")
    print("Training on device:", agent.device)

    # Initialize TensorBoard writer
    writer = SummaryWriter("logs")


    n_games = 100000

    best_score = -np.inf
    episode_identifier = (
        f"actor_lr={actor_learning_rate} critic_lr={critic_learning_rate} "
        f"batch_size={batch_size} layer1={layer1_size} layer2={layer2_size} "
        f"tau=0.005 env={env_name} {int(time.time())}"
    )

    # Optional: Load pretrained models if you have them
    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            # Agent chooses an action
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward

            # Store and learn
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()

            observation = next_observation
            

        # Log the score in TensorBoard
        writer.add_scalar(f"score/{episode_identifier}", score, global_step=i)

        # Save models every 10 episodes
        if i % 10 == 0:
            agent.save_models()
        
        print(f"Episode {i}, Score {score}")