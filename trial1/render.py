import os
import gym
import torch
import numpy as np
import cv2
from collections import deque
from train import Agent, args

def create_video(source, fps=60, output_name='output'):
    print("[create_video] Starting video creation.")
    out = cv2.VideoWriter(
        output_name + '.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (source[0].shape[1], source[0].shape[0])
    )
    for i, frame in enumerate(source):
        if i % 100 == 0:
            print(f"[create_video] Writing frame {i} of {len(source)}")
        out.write(frame)
    out.release()
    print("[create_video] Video creation complete.")

def preprocess(frame):
    # Convert the RGB frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to (96,96) if needed
    resized = cv2.resize(gray, (96, 96))
    return resized

if __name__ == "__main__":
    env = gym.make('CarRacing-v2', render_mode="human")
    output_folder = r"C:\Users\ronki\OneDrive\Documents\GitHub\pytorch_car_caring\videos"
    os.makedirs(output_folder, exist_ok=True)
    frames = []
    
    agent = Agent()
    agent.net.load_state_dict(torch.load(
        'param/ppo_net_params.pkl',
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ))
    
    # Use a deque to hold the last img_stack preprocessed frames
    frame_stack = deque(maxlen=args.img_stack)
    
    # Reset env, preprocess the initial frame, and fill the stack
    print("[Main] Resetting environment and initializing frame stack")
    raw_state, _ = env.reset()  # raw_state shape: (96,96,3)
    processed = preprocess(raw_state)  # shape: (96,96)
    for _ in range(args.img_stack):
        frame_stack.append(processed)
        
    # Combine stack into a numpy array with shape (img_stack, 96,96)
    state = np.stack(frame_stack, axis=0)
    
    num_episodes = 1
    episode = 0
    step_count = 0
    while episode < num_episodes:
        # The network might expect a batch dimension; add one if needed:
        action, _ = agent.select_action(state)
        print(f"[Episode {episode}] Step {step_count}")
        
        scaled_action = action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
        
        raw_state, reward, done, truncated, _ = env.step(scaled_action)
        
        frame = env.render()  # capture the rendered frame (for video saving)
        
        frames.append(frame)
        
        # Process new frame for state update
        processed = preprocess(raw_state)
        
        frame_stack.append(processed)
        state = np.stack(frame_stack, axis=0)
        
        step_count += 1
        
        if done or truncated:
            raw_state, _ = env.reset()
            processed = preprocess(raw_state)
            frame_stack = deque([processed]*args.img_stack, maxlen=args.img_stack)
            state = np.stack(frame_stack, axis=0)
            episode += 1
            step_count = 0
    
    video_output_path = os.path.join(output_folder, "recorded_video")
    create_video(frames, fps=60, output_name=video_output_path)
    env.close()
