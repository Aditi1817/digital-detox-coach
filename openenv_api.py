"""
OpenEnv API Server for Digital Detox Coach
Provides the required endpoints for the hackathon submission
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from environment import DigitalDetoxEnv
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store active environments
environments = {}

@app.route('/api/openenv/reset', methods=['POST', 'GET', 'OPTIONS'])
def reset_environment():
    """Reset the environment - Required by OpenEnv"""
    try:
        # Handle OPTIONS preflight
        if request.method == 'OPTIONS':
            return '', 200
            
        # Get environment ID from request
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = {}
            
        env_id = data.get('env_id', 'default')
        
        # Create or reset environment
        if env_id not in environments:
            environments[env_id] = DigitalDetoxEnv()
        
        env = environments[env_id]
        state = env.reset()
        
        return jsonify({
            'success': True,
            'state': state.tolist(),
            'observation_space': {
                'shape': list(env.observation_space.shape),
                'low': env.observation_space.low.tolist(),
                'high': env.observation_space.high.tolist()
            },
            'action_space': {
                'n': env.action_space.n
            }
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/openenv/step', methods=['POST', 'OPTIONS'])
def step_environment():
    """Take a step in the environment - Required by OpenEnv"""
    try:
        # Handle OPTIONS preflight
        if request.method == 'OPTIONS':
            return '', 200
            
        data = request.get_json()
        env_id = data.get('env_id', 'default')
        action = data.get('action')
        
        if action is None:
            return jsonify({'success': False, 'error': 'action required'}), 400
            
        if env_id not in environments:
            return jsonify({'success': False, 'error': 'Environment not initialized. Call reset first.'}), 400
        
        env = environments[env_id]
        next_state, reward, done, info = env.step(action)
        
        return jsonify({
            'success': True,
            'next_state': next_state.tolist(),
            'reward': float(reward),
            'done': bool(done),
            'info': info
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/openenv/validate', methods=['GET', 'POST', 'OPTIONS'])
def validate_environment():
    """Validate environment meets OpenEnv standards"""
    try:
        # Handle OPTIONS preflight
        if request.method == 'OPTIONS':
            return '', 200
            
        # Test environment creation
        test_env = DigitalDetoxEnv()
        
        # Test reset
        state = test_env.reset()
        
        # Test step
        action = test_env.action_space.sample()
        next_state, reward, done, info = test_env.step(action)
        
        # Test observation space
        assert test_env.observation_space.contains(state), "State not in observation space"
        
        return jsonify({
            'success': True,
            'valid': True,
            'message': 'Environment is OpenEnv compatible',
            'details': {
                'observation_shape': list(test_env.observation_space.shape),
                'action_space_size': test_env.action_space.n,
                'test_step_worked': True
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'valid': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/api/openenv/info', methods=['GET'])
def get_env_info():
    """Get environment information"""
    try:
        env = DigitalDetoxEnv()
        return jsonify({
            'name': 'AI Digital Detox Coach',
            'version': '1.0.0',
            'description': 'RL-based digital wellness coach',
            'observation_space': {
                'type': 'Box',
                'shape': list(env.observation_space.shape),
                'low': env.observation_space.low.tolist(),
                'high': env.observation_space.high.tolist()
            },
            'action_space': {
                'type': 'Discrete',
                'n': env.action_space.n,
                'actions': ['allow_app', 'block_app', 'suggest_study', 'suggest_break', 'suggest_sleep']
            },
            'max_episode_steps': env.max_steps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'digital-detox-coach'})

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'service': 'AI Digital Detox Coach',
        'endpoints': {
            'reset': '/api/openenv/reset',
            'step': '/api/openenv/step',
            'validate': '/api/openenv/validate',
            'info': '/api/openenv/info',
            'health': '/health'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)