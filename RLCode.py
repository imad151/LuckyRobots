import gym
import numpy as np
import cv2 
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import torch

class RoboticArmEnv(gym.Env):
    """
    Observations are file paths to RGB and Depth images.
    Actions are represented as a 6x1 matrix for 6-DOF movements.
    """
    def __init__(self, rgb_image_path, depth_image_path):
        super(RoboticArmEnv, self).__init__()
        
        # Input will be 4 channels: 3 for RGB and 1 for Depth image
        self.observation_shape = (4, 64, 64)  # Placeholder resolution
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
        
        # Action space: 6-DOF motion represented as a 6x1 matrix
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6, 1), dtype=np.float32
        )
        
        # File paths for the RGB and Depth images
        self.rgb_image_path = rgb_image_path
        self.depth_image_path = depth_image_path

        # Placeholder for the internal state
        self.state = None
        self.reset()

    def reset(self):
        """
        Reset the environment state and return the initial observation.
        """
        return self._get_observation()

    def step(self, action):
        """
        Execute the given action, return observation, reward, done, and info.
        """

        # Get the next observation
        obs = self._get_observation()

        # Calculate reward based on the current action and state
        reward = self._calculate_reward(action)

        # Define when the episode is done
        done = False 

        return obs, reward, done

    def _get_observation(self):
        """
        Load and preprocess the RGB and Depth images from file paths.
        """
        rgb_image = cv2.imread(self.rgb_image_path)
        rgb_image = cv2.resize(rgb_image, (64, 64))  # Resize
        rgb_image = rgb_image.transpose(2, 0, 1)  # Convert to [C, H, W] format

        # Load the Depth image
        depth_image = cv2.imread(self.depth_image_path, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.resize(depth_image, (64, 64))  # Resize
        depth_image = np.expand_dims(depth_image, axis=0)  # Add channel dimension

        # Combine RGB and Depth into a single observation
        observation = np.concatenate([rgb_image, depth_image], axis=0)
        return observation

    def _calculate_reward(self, action):
        """
        Compute the reward based on the action and the environment's current state.
        """
        # Placeholder
        return np.random.random()

    def render(self, mode='human'):
        """
        Render the environment for debugging.
        """
        pass 

    def close(self):
        pass

rgb_image_path = "path_to_rgb_image.png"
depth_image_path = "path_to_depth_image.png"
env = RoboticArmEnv(rgb_image_path, depth_image_path)

check_env(env, warn=True)

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    gradient_steps=1,
    verbose=1
)

model.learn(total_timesteps=10000)

model.save("robotic_arm_dqn")

# model = DQN.load("robotic_arm_dqn")

# Testing the model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
