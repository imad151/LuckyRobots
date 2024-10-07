import gym
from stable_baselines3 import PPO

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 10000 time steps
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model (optional)
# model = PPO.load("ppo_cartpole")

# Evaluate the trained agent
episodes = 5
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
env.close()
