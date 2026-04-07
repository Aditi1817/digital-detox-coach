"""
Digital Detox Coach - Inference Module
Meets all submission requirements for OpenEnv
"""

import os
import sys
import json
import requests
import numpy as np
from openai import OpenAI
from typing import Dict, Any, List, Optional

# ============================================
# ENVIRONMENT VARIABLES (as required by checklist)
# ============================================

# Required - with defaults allowed
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Required - NO DEFAULT! Must come from environment
HF_TOKEN = os.getenv("HF_TOKEN")  # No default value here!

# Optional - only if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

# Your Space URL
SPACE_URL = os.getenv("SPACE_URL", "https://aditishedbale-digital-detox-coach.hf.space")

# ============================================
# OPENAI CLIENT (as required)
# ============================================

# Initialize OpenAI client with the configured variables
client = OpenAI(
    api_key=HF_TOKEN,  # Use HF_TOKEN as API key
    base_url=API_BASE_URL
)

# ============================================
# DIGITAL DETOX API CLIENT
# ============================================

class DigitalDetoxClient:
    """Client for interacting with the Digital Detox Coach Space"""
    
    def __init__(self, space_url: str = SPACE_URL):
        self.space_url = space_url
        self.observation = None
        self.total_reward = 0
        self.step_count = 0
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        print(f"STEP: Calling reset endpoint at {self.space_url}/reset")
        
        try:
            response = requests.post(
                f"{self.space_url}/reset",
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.observation = data.get("observation")
                self.total_reward = 0
                self.step_count = 0
                print(f"STEP: Reset successful - Observation: {self.observation}")
                return data
            else:
                raise Exception(f"Reset failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"ERROR: Reset failed - {str(e)}")
            raise
    
    def take_action(self, action: int) -> Dict[str, Any]:
        """Take an action in the environment"""
        print(f"STEP: Taking action {action} at {self.space_url}/action")
        
        try:
            response = requests.post(
                f"{self.space_url}/action",
                headers={"Content-Type": "application/json"},
                json={"action": action},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.observation = data.get("next_state")
                self.total_reward += data.get("reward", 0)
                self.step_count += 1
                print(f"STEP: Action {action} complete - Reward: {data.get('reward', 0):.2f}, Total: {self.total_reward:.2f}")
                return data
            else:
                raise Exception(f"Action failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"ERROR: Action failed - {str(e)}")
            raise
    
    def get_agent_action(self, state: Optional[List[float]] = None, training: bool = False) -> int:
        """Get action from the DQN agent"""
        if state is None:
            state = self.observation
            
        print(f"STEP: Getting agent action for state: {state}")
        
        try:
            response = requests.post(
                f"{self.space_url}/agent_action",
                headers={"Content-Type": "application/json"},
                json={"state": state, "training": training},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                action = data.get("action")
                print(f"STEP: Agent chose action {action} (epsilon: {data.get('epsilon', 0):.3f})")
                return action
            else:
                # Fallback to random action if agent endpoint fails
                import random
                action = random.randint(0, 4)
                print(f"STEP: Agent endpoint failed, using random action: {action}")
                return action
                
        except Exception as e:
            print(f"WARNING: Agent action failed - {str(e)}, using random action")
            import random
            return random.randint(0, 4)

# ============================================
# LLM-BASED DECISION MAKING (Optional enhancement)
# ============================================

class DigitalDetoxLLM:
    """LLM-based decision maker using OpenAI client"""
    
    def __init__(self):
        self.client = client
        self.model = MODEL_NAME
        
    def decide_action(self, observation: List[float]) -> int:
        """Use LLM to decide the best action based on current observation"""
        
        screen_time, study_time, sleep_time, productivity, social_media = observation
        
        prompt = f"""
        You are a Digital Detox Coach. Based on the current user state, choose the best action.
        
        Current State:
        - Screen Time: {screen_time:.1f} hours
        - Study Time: {study_time:.1f} hours  
        - Sleep Time: {sleep_time:.1f} hours
        - Productivity: {productivity:.0f}%
        - Social Media Usage: {social_media:.1f} hours
        
        Available Actions:
        0: Allow App (let user use apps)
        1: Block App (block distracting apps)
        2: Suggest Study (encourage studying)
        3: Suggest Break (recommend taking a break)
        4: Suggest Sleep (recommend sleeping)
        
        Consider:
        - If screen time > 4h, prefer blocking apps (action 1)
        - If productivity < 50%, prefer suggesting study (action 2)
        - If sleep < 7h, prefer suggesting sleep (action 4)
        - If screen time > 6h and productivity < 40%, take strict actions
        
        Respond with ONLY the action number (0-4), nothing else.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful digital detox coach. Respond with only a number 0-4."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            action_text = response.choices[0].message.content.strip()
            action = int(action_text)
            
            if 0 <= action <= 4:
                print(f"STEP: LLM decided action: {action}")
                return action
            else:
                print(f"WARNING: LLM returned invalid action {action}, using fallback")
                return self._fallback_action(observation)
                
        except Exception as e:
            print(f"WARNING: LLM decision failed - {str(e)}, using fallback")
            return self._fallback_action(observation)
    
    def _fallback_action(self, observation: List[float]) -> int:
        """Rule-based fallback when LLM fails"""
        screen_time, productivity, sleep_time = observation[0], observation[3], observation[2]
        
        if screen_time > 6:
            return 1  # Block app
        elif screen_time > 4:
            return 0  # Allow with caution
        elif productivity < 50:
            return 2  # Suggest study
        elif sleep_time < 7:
            return 4  # Suggest sleep
        else:
            return 3  # Suggest break

# ============================================
# MAIN INFERENCE FUNCTION
# ============================================

def run_inference(num_steps: int = 10, use_llm: bool = False) -> Dict[str, Any]:
    """
    Main inference function with structured logging (START/STEP/END format)
    
    Args:
        num_steps: Number of steps to run
        use_llm: Whether to use LLM for decision making
    
    Returns:
        Dictionary with inference results
    """
    
    # START log - REQUIRED format
    print("START: Digital Detox Coach Inference")
    
    # Check required environment variables
    if HF_TOKEN is None:
        print("ERROR: HF_TOKEN environment variable is not set")
        print("ERROR: Please set HF_TOKEN in your Space secrets")
        sys.exit(1)
    
    print(f"STEP: Environment check passed - API_BASE_URL: {API_BASE_URL}, MODEL_NAME: {MODEL_NAME}")
    
    # Initialize client
    print("STEP: Initializing Digital Detox Client")
    detox_client = DigitalDetoxClient()
    
    # Initialize LLM if requested
    llm = None
    if use_llm:
        print("STEP: Initializing LLM decision maker")
        llm = DigitalDetoxLLM()
    
    # Reset environment
    print("STEP: Resetting environment")
    reset_result = detox_client.reset()
    
    results = {
        "success": True,
        "steps": [],
        "total_reward": 0,
        "final_state": None
    }
    
    # Run inference loop
    print(f"STEP: Starting inference loop for {num_steps} steps")
    
    for step in range(num_steps):
        print(f"STEP: Step {step + 1}/{num_steps}")
        
        # Get current observation
        current_state = detox_client.observation
        print(f"STEP: Current state - Screen: {current_state[0]:.1f}h, Productivity: {current_state[3]:.0f}%")
        
        # Decide action
        if use_llm and llm:
            action = llm.decide_action(current_state)
        else:
            action = detox_client.get_agent_action(current_state, training=False)
        
        # Take action
        step_result = detox_client.take_action(action)
        
        # Store step result
        results["steps"].append({
            "step": step + 1,
            "action": action,
            "reward": step_result.get("reward", 0),
            "next_state": step_result.get("next_state"),
            "done": step_result.get("done", False)
        })
        
        print(f"STEP: Step {step + 1} complete - Action: {action}, Reward: {step_result.get('reward', 0):.2f}")
        
        # Check if episode is done
        if step_result.get("done", False):
            print(f"STEP: Episode done at step {step + 1}")
            break
    
    # Finalize results
    results["total_reward"] = detox_client.total_reward
    results["final_state"] = detox_client.observation
    results["total_steps"] = detox_client.step_count
    
    print(f"STEP: Inference complete - Total steps: {results['total_steps']}, Total reward: {results['total_reward']:.2f}")
    
    # END log - REQUIRED format
    print("END: Digital Detox Coach Inference")
    
    return results

# ============================================
# COMMAND LINE ENTRY POINT
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Digital Detox Coach Inference")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for decision making")
    parser.add_argument("--test-api", action="store_true", help="Test API connection only")
    
    args = parser.parse_args()
    
    if args.test_api:
        # Test API connection
        print("START: API Connection Test")
        try:
            response = requests.get(f"{SPACE_URL}/health", timeout=10)
            if response.status_code == 200:
                print(f"STEP: API is healthy - {response.json()}")
                print("END: API Connection Test - SUCCESS")
            else:
                print(f"ERROR: API returned {response.status_code}")
                print("END: API Connection Test - FAILED")
        except Exception as e:
            print(f"ERROR: Cannot connect to API - {e}")
            print("END: API Connection Test - FAILED")
    else:
        # Run full inference
        results = run_inference(num_steps=args.steps, use_llm=args.use_llm)
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        print(json.dumps(results, indent=2))
        print("="*50)
