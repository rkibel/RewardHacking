import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class CarRacingSpeedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Get speed from the underlying Car object if available
        if hasattr(self.env.unwrapped, 'car') and hasattr(self.env.unwrapped.car.hull, 'linearVelocity'):
            velocity = self.env.unwrapped.car.hull.linearVelocity
            speed = np.linalg.norm([velocity.x, velocity.y])
            info['speed'] = speed
        return obs, reward, done, truncated, info

class HackyCarRacing(CarRacingSpeedWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        speed_bonus = info.get('speed', 0.0) * 3.0

        # Simple grass detection: count green pixels in the frame
        green_channel = obs[..., 1]
        is_grass = (green_channel > 150)  # tune threshold if needed
        grass_ratio = is_grass.mean()
        grass_penalty = grass_ratio * 100.0  # adjust penalty scaling

        print(speed_bonus, grass_penalty)
        reward += speed_bonus
        reward -= grass_penalty
        return obs, reward, done, truncated, info

def main():
    env = HackyCarRacing(gym.make('CarRacing-v3', render_mode="human"))
    model = PPO.load("./checkpoints/car_racing_500000_steps", env=env)
    checkpoint_callback = CheckpointCallback(save_freq=1000, 
                                             save_path='./hacky3_checkpoints/',
                                             name_prefix='car_racing')
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

if __name__ == "__main__":
    main()
