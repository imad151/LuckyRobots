import gym
import numpy as np
import cv2  # For image manipulation
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

class RoboticArmEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The observation is composed of:
    - RGB camera image (shape: [3, 64, 64])
    - Depth camera image (shape: [1, 64, 64])
    
    The action space consists of 6-DOF movements.
    """
    def __init__(self):
        super(RoboticArmEnv, self).__init__()
        
        self.observation_shape = (4, 64, 64)  # 3 RGB channels + 1 Depth channel
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
        
        # Action space: 6-DOF motion (-1 to 1 range)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Placeholder for environment state (e.g., robot's internal state)
        self.state = None
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        # Reset state and generate initial observation
        self.state = np.zeros(self.observation_shape, dtype=np.uint8)
        return self._get_observation()

    def step(self, action):
        """
        Execute the given action and return the result.
        """

        # self.state = np.random.randint(0, 255, self.observation_shape, dtype=np.uint8)
        
        reward = self._calculate_reward(action)
        
        done = False  # Set to True if episode should terminate
        
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """
        Get the current observation (concatenated RGB and Depth image).
        """
        return self.state

    def _calculate_reward(self, action):
        """
        Calculate reward based on the action.
        """
        # returning random for now
        return np.random.random()

    def render(self, mode='human'):
        """
        Render the environment (optional).
        """
        if mode == 'human':
            rgb_image = self.state[:3, :, :]
            cv2.imshow("RGB Image", rgb_image)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

env = RoboticArmEnv()

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=0.01,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    gradient_steps=1,
    verbose=1
)

model.learn(total_timesteps=10000)

model.save("robotic_arm_dqn")

# Testing the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
