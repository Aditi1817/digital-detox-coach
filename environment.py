"""
Digital Detox Environment - OpenEnv Compatible
Simulates digital behavior and provides rewards for healthy habits
"""

import numpy as np
from typing import Tuple, Dict, Any
import gym
from gym import spaces
import random

class DigitalDetoxEnv(gym.Env):
    """
    Custom Environment for Digital Detox Coach
    Teaches optimal digital behavior through reinforcement learning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(DigitalDetoxEnv, self).__init__()
        
        # Default configuration
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 24)  # 24 hours simulation
        self.current_step = 0
        
        # Define action space (5 actions)
        # 0: allow_app, 1: block_app, 2: suggest_study, 3: suggest_break, 4: suggest_sleep
        self.action_space = spaces.Discrete(5)
        
        # Define observation space (5 features)
        # [screen_time, study_time, sleep_time, productivity_score, social_media_usage]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([24, 12, 12, 100, 24], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.daily_reward = 0
        self.history = {
            'screen_time': [],
            'study_time': [],
            'sleep_time': [],
            'productivity': [],
            'social_media': [],
            'rewards': []
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.daily_reward = 0
        
        # Initialize state with healthy values
        self.state = np.array([
            random.uniform(1, 3),  # screen_time (hours so far)
            random.uniform(0, 1),  # study_time
            random.uniform(6, 8),  # sleep_time (previous night)
            random.uniform(50, 70), # productivity_score
            random.uniform(0, 1)   # social_media_usage
        ], dtype=np.float32)
        
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        Returns: next_state, reward, done, info
        """
        self.current_step += 1
        
        # Get current state variables
        screen_time, study_time, sleep_time, productivity, social_media = self.state
        
        # Apply action effects
        reward = 0
        info = {'action_taken': action}
        
        # Action mapping
        actions = {
            0: "allow_app",
            1: "block_app", 
            2: "suggest_study",
            3: "suggest_break",
            4: "suggest_sleep"
        }
        
        action_name = actions[action]
        info['action_name'] = action_name
        
        # Update state based on action
        if action_name == "allow_app":
            # Allowing app increases screen time and social media usage
            screen_time = min(24, screen_time + random.uniform(0.5, 1.5))
            social_media = min(24, social_media + random.uniform(0.5, 1))
            productivity = max(0, productivity - random.uniform(5, 15))
            reward -= 5  # Negative reward for allowing app
            
        elif action_name == "block_app":
            # Blocking app reduces screen time and social media
            screen_time = max(0, screen_time - random.uniform(0.5, 1))
            social_media = max(0, social_media - random.uniform(0.5, 1))
            productivity = min(100, productivity + random.uniform(5, 10))
            reward += 10  # Positive reward for blocking apps
            
        elif action_name == "suggest_study":
            # Study suggestion improves productivity
            study_time = min(12, study_time + random.uniform(0.5, 1))
            productivity = min(100, productivity + random.uniform(10, 20))
            reward += 15  # High positive reward for studying
            
        elif action_name == "suggest_break":
            # Break reduces screen time slightly
            screen_time = max(0, screen_time - random.uniform(0.2, 0.5))
            productivity = min(100, productivity + random.uniform(5, 10))
            reward += 5  # Moderate reward for taking breaks
            
        elif action_name == "suggest_sleep":
            # Sleep suggestion improves sleep time
            sleep_time = min(12, sleep_time + random.uniform(0.5, 1))
            reward += 20  # High reward for good sleep
        
        # Natural dynamics (time passing)
        screen_time += random.uniform(0.1, 0.3)
        study_time = max(0, study_time - random.uniform(0.1, 0.2))
        productivity = max(0, productivity - random.uniform(2, 5))
        social_media += random.uniform(0.1, 0.3) if social_media > 0 else 0
        
        # Reward calculations based on current state
        # Less screen time = positive reward
        if screen_time < 4:
            reward += 10
        elif screen_time > 8:
            reward -= 15
        
        # More study = positive reward
        if study_time > 4:
            reward += 15
        elif study_time < 1:
            reward -= 10
        
        # Good sleep = positive reward
        if sleep_time >= 7 and sleep_time <= 9:
            reward += 20
        elif sleep_time < 6:
            reward -= 15
        
        # Excessive social media = negative reward
        if social_media > 3:
            reward -= 20
        elif social_media < 1:
            reward += 10
        
        # Late night usage (after 10 PM)
        if self.current_step >= 22 and screen_time > 0:
            reward -= 25
        
        # Update state
        self.state = np.array([
            max(0, min(24, screen_time)),
            max(0, min(12, study_time)),
            max(0, min(12, sleep_time)),
            max(0, min(100, productivity)),
            max(0, min(24, social_media))
        ], dtype=np.float32)
        
        # Store history
        self.history['screen_time'].append(self.state[0])
        self.history['study_time'].append(self.state[1])
        self.history['sleep_time'].append(self.state[2])
        self.history['productivity'].append(self.state[3])
        self.history['social_media'].append(self.state[4])
        self.history['rewards'].append(reward)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Accumulate daily reward
        self.daily_reward += reward
        
        if done:
            # Final reward for overall health
            final_reward = 0
            if self.state[0] < 5:  # Screen time less than 5 hours
                final_reward += 50
            if self.state[1] > 4:  # Study time more than 4 hours
                final_reward += 50
            if 7 <= self.state[2] <= 9:  # Sleep within healthy range
                final_reward += 75
            if self.state[3] > 80:  # High productivity
                final_reward += 50
            if self.state[4] < 2:  # Low social media
                final_reward += 50
                
            reward += final_reward
            info['final_bonus'] = final_reward
            info['total_reward'] = self.daily_reward + final_reward
        
        return self.state, reward, done, info
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            screen_time, study_time, sleep_time, productivity, social_media = self.state
            print(f"\n=== Hour {self.current_step} ===")
            print(f"Screen Time: {screen_time:.1f}h")
            print(f"Study Time: {study_time:.1f}h")
            print(f"Sleep Time: {sleep_time:.1f}h")
            print(f"Productivity: {productivity:.1f}%")
            print(f"Social Media: {social_media:.1f}h")
            print(f"Daily Reward: {self.daily_reward:.1f}")
    
    def get_history(self):
        """Return training history"""
        return self.history