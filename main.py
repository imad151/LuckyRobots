from stable_baselines3 import TD3
from RLCode import RoboticArmEnv

import os

import luckyrobots as lr

class SimulationController:
    def __init__(self):
        self.env = RoboticArmEnv()
        
        self.model = TD3("CnnPolicy", self.env, verbose=1, buffer_size=10000)

    def update_environment(self, rgb_image, depth_image):
        """
        Send RGB and Depth images to the Gym environment.
        """
        self.env.update_observation(rgb_image, depth_image)

    def predict_action(self):
        """
        Get the predicted action from the model.
        """
        obs = self.env.state  # Get current state from the Gym environment
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def execute_action(self, action):
        """
        Execute the predicted action in the simulation.
        ach row of the action matrix corresponds to a joint with specific value ranges.
        """
        commands = []

        # Map action values to joint commands
        commands.append([f"EX1 {int(action[0, 0] * 180)}"])  # EX1: -180 to +180
        commands.append([f"EX2 {int(action[1, 0] * 180)}"])  # EX2: -180 to +180
        commands.append([f"EX3 {int(action[2, 0] * 180)}"])  # EX3: -180 to +180
        commands.append([f"EX4 {int(action[3, 0] * 360)}"])  # EX4: -360 to +360
        commands.append([f"U {int(action[4, 0] * 10)}"])      # U: -10 to +10
        commands.append([f"G {int(action[5, 0] * 10)}"])      # G: -10 to +10

        lr.send_message(commands)

    def train(self, total_timesteps):
        """
        Train the DQN model.
        """
        self.model.learn(total_timesteps=total_timesteps)

# Example usage
if __name__ == '__main__':
    sim = SimulationController()
    lr.start()

    @lr.on("robot_output")
    def get_image_path(msg):
        if msg and isinstance(msg, dict) and 'rgb_cam1' in msg:
            image_path = msg['rgb_cam2'].get('file_path')
            greyscale_img = msg['depth_cam2'].get('file_path')

            if os.path.exists(image_path) and os.path.exists(greyscale_img):
                rgb_image = image_path
                depth_image = greyscale_img




        sim.update_environment(rgb_image, depth_image)
        action = sim.predict_action()
        sim.execute_action(action)
