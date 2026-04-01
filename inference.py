"""
Digital Detox Coach - Inference Script
Must follow [START], [STEP], [END] logging format
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from environment import DigitalDetoxEnv
from agent import DQNAgent

# Get environment variables (required by hackathon)
API_BASE_URL = os.environ.get('API_BASE_URL', '')
MODEL_NAME = os.environ.get('MODEL_NAME', 'digital-detox-coach')
HF_TOKEN = os.environ.get('HF_TOKEN', '')

class DigitalDetoxInference:
    def __init__(self):
        """Initialize environment and agent"""
        self.env = DigitalDetoxEnv({'max_steps': 24})
        self.agent = DQNAgent(state_size=5, action_size=5)
        
        # Try to load trained model
        try:
            self.agent.load('results/final_detox_agent.pth')
            self.model_loaded = True
        except:
            self.model_loaded = False
        
        self.total_reward = 0
        self.step_count = 0
        
    def reset(self):
        """Reset environment and return initial state"""
        state = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        return state
    
    def step(self, action):
        """Take action and return results"""
        next_state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        
        # Get action name
        action_names = ["allow_app", "block_app", "suggest_study", "suggest_break", "suggest_sleep"]
        action_name = action_names[action]
        
        return {
            'state': next_state.tolist(),
            'reward': float(reward),
            'done': done,
            'total_reward': float(self.total_reward),
            'action_taken': action_name,
            'screen_time': float(next_state[0]),
            'study_time': float(next_state[1]),
            'sleep_time': float(next_state[2]),
            'productivity': float(next_state[3]),
            'social_media': float(next_state[4]),
            'step': self.step_count,
            'info': info
        }
    
    def get_action(self, state):
        """Get action from agent (or random if no model)"""
        if self.model_loaded:
            return self.agent.act(state, training=False)
        else:
            # Smart fallback: prioritize blocking apps and suggesting study
            # If screen time > 3h, block app
            if state[0] > 3:
                return 1  # block_app
            # If study time < 2h, suggest study
            elif state[1] < 2:
                return 2  # suggest_study
            # If late night (> 10 PM) and screen time > 0
            elif self.step_count >= 22 and state[0] > 0:
                return 4  # suggest_sleep
            # Default: suggest break
            else:
                return 3  # suggest_break


def main():
    """Main inference function with required logging format"""
    
    # Initialize
    inference = DigitalDetoxInference()
    num_episodes = 3  # Run 3 episodes for evaluation
    
    # Required: [START] log format
    print(f"[START] Digital Detox Coach Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Episodes: {num_episodes}")
    print(f"Model Loaded: {inference.model_loaded}")
    print("-" * 50)
    
    all_results = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = inference.reset()
        episode_reward = 0
        episode_steps = []
        
        print(f"\n[STEP] Episode {episode + 1} - Reset")
        print(json.dumps({
            'episode': episode + 1,
            'initial_state': {
                'screen_time': float(state[0]),
                'study_time': float(state[1]),
                'sleep_time': float(state[2]),
                'productivity': float(state[3]),
                'social_media': float(state[4])
            }
        }))
        
        # Run episode
        for step in range(24):
            # Get action from agent
            action = inference.get_action(state)
            
            # Take step
            result = inference.step(action)
            
            # Required: [STEP] log format for each step
            print(f"\n[STEP] Episode {episode + 1}, Step {step + 1}")
            print(json.dumps({
                'step': step + 1,
                'action': result['action_taken'],
                'reward': result['reward'],
                'total_reward': result['total_reward'],
                'screen_time': result['screen_time'],
                'study_time': result['study_time'],
                'productivity': result['productivity'],
                'social_media': result['social_media'],
                'done': result['done']
            }))
            
            state = result['state']
            episode_reward = result['total_reward']
            episode_steps.append(result)
            
            if result['done']:
                break
        
        # Store episode results
        all_results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'final_screen_time': result['screen_time'],
            'final_study_time': result['study_time'],
            'final_productivity': result['productivity'],
            'final_social_media': result['social_media'],
            'steps': len(episode_steps)
        })
        
        print(f"\n[STEP] Episode {episode + 1} - Complete")
        print(json.dumps({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'final_state': {
                'screen_time': result['screen_time'],
                'study_time': result['study_time'],
                'productivity': result['productivity'],
                'social_media': result['social_media']
            }
        }))
    
    # Required: [END] log format with summary
    print("\n" + "=" * 50)
    print("[END] Inference Complete")
    print(json.dumps({
        'total_episodes': num_episodes,
        'average_reward': float(np.mean([r['total_reward'] for r in all_results])),
        'average_screen_time': float(np.mean([r['final_screen_time'] for r in all_results])),
        'average_study_time': float(np.mean([r['final_study_time'] for r in all_results])),
        'average_productivity': float(np.mean([r['final_productivity'] for r in all_results])),
        'success_rate': float(len([r for r in all_results if r['final_screen_time'] < 4]) / num_episodes),
        'model_loaded': inference.model_loaded,
        'timestamp': datetime.now().isoformat()
    }))
    print("=" * 50)
    
    # Return success code
    return 0


if __name__ == "__main__":
    sys.exit(main())