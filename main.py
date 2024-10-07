import numpy as np
import os
import torch
from luckyrobots import core as lr
from PIL import Image
from stable_baselines3 import PPO

class BasicMovementRL:
    def __init__(self, model, temperature=1.0, bin_size=1.0):
        self.model = model
        self.temperature = temperature
        self.bin_size = bin_size
        self.state_visit_counts = {}
        self.max_bin_count = 100
        self.observations = np.zeros(5)  # Example initialization
        self.hit_count = 0
        self.reward = 0

    def GenerateCommand(self):
        action_probs = self.GetActionProbabilities(self.observations)
        action = np.random.choice(len(action_probs), p=action_probs)
        direction, speed = self.MapActionToCommand(action)
        self.MoveRobot(action, direction, speed)

    def GetActionProbabilities(self, observations):
        with torch.no_grad():
            action, _ = self.model.predict(observations, deterministic=True)
        action_probs = self.softmax(action, self.temperature)
        return action_probs

    def softmax(self, x, temperature):
        x = x / temperature
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def MapActionToCommand(self, action):
        if action == 0:
            return 360, 1
        elif action == 1:
            return 270, 1
        elif action == 2:
            return 180, 1
        elif action == 3:
            return 90, 1

    def UpdateModel(self, total_timesteps=100000):
        # Training loop for the model
        self.GenerateCommand()  # Generate commands based on current observations
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.model.save("BasicMovementsModel.zip")

    @lr.on("robot_output")
    def GetObservations(self, message: dict):
        # Process observations from the message
        body_pos = [float(message["body_pos"]["contents"].get(k, 0)) for k in ("tx", "ty", "tz")]
        depthcam_values = [self.ProcessDepthCam(message["depth_cam1"]["file_path"]),
                        self.ProcessDepthCam(message["depth_cam2"]["file_path"])]
        self.observations = np.array(body_pos + depthcam_values)
        discretized_state = self.DiscretizeState(self.observations)
        self.UpdateVisitCounts(discretized_state)


    def ProcessDepthCam(self, img_path: str) -> float:
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                greyscale = np.array(img.convert('L'))
                proximity_metric = np.mean(greyscale)
                return (255 - proximity_metric) / 255 * 10.0
        return 0.0
        
    def DiscretizeState(self, state):
        state_bins = np.floor_divide(state, self.bin_size)
        return tuple(state_bins)

    def UpdateVisitCounts(self, state):
        if state not in self.state_visit_counts:
            self.state_visit_counts[state] = 0
        self.state_visit_counts[state] += 1

    @lr.on("hit_count")
    def GetHitCount(self, hits):
        self.hit_count = hits

    @lr.on("task_complete")
    def IsDone(self, status):
        return status

def call():
    from CreateEnvironment import ToEnv
    lr.start()
    model_path = "BasicMovementsModel.zip"

    if os.path.exists(model_path):
        model = PPO.load(model_path, env=ToEnv())
    else:
        env = ToEnv()
        policy_kwagr = dict(net_arch=[128, 128, 128])
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwagr, verbose=1)

    basic_movements = BasicMovementRL(model)
    total_timesteps = 100000
    for _ in total_timesteps:
        basic_movements.UpdateModel(total_timesteps=total_timesteps)

    model.save("BaicMovementsModel.zip")

if __name__ == '__main__':
    call()