import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback

# Import your actual classes
from agent import DQNAgent
from environment import DigitalDetoxEnv

app = Flask(__name__)
CORS(app)  # CRITICAL - enables POST requests

PORT = int(os.environ.get('PORT', 7860))

# Initialize agent and environment
state_size = 5  # Matches your observation space (5 features)
action_size = 5  # Matches your action space (5 actions)

print("=" * 50)
print("Initializing Digital Detox Coach...")
print("=" * 50)

# Initialize DQN Agent
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=32,
    memory_size=10000,
    target_update=10
)

# Initialize Environment
env = DigitalDetoxEnv()

# Try to load pretrained model if exists
try:
    agent.load('final_detox_agent.pth')
    print("✓ Loaded pretrained model from final_detox_agent.pth")
except Exception as e:
    print(f"⚠ No pretrained model found: {e}")
    print("  Using fresh agent with random weights")

print("✓ Agent and Environment initialized successfully")
print(f"  State size: {state_size}")
print(f"  Action size: {action_size}")
print(f"  Device: {agent.device}")
print("=" * 50)

# ============================================
# WEB INTERFACE - HTML/CSS/JavaScript
# ============================================

def get_interface_html():
    """Return the HTML interface for the Digital Detox Coach"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digital Detox Coach</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .metric-title {
                font-size: 14px;
                color: #555;
                margin-bottom: 10px;
                font-weight: bold;
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
            
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                transition: width 0.5s ease;
                border-radius: 10px;
            }
            
            .progress-fill.good { background: linear-gradient(90deg, #4CAF50, #8BC34A); }
            .progress-fill.warning { background: linear-gradient(90deg, #FF9800, #FFC107); }
            .progress-fill.danger { background: linear-gradient(90deg, #f44336, #FF5722); }
            
            .actions-section {
                margin: 30px 0;
            }
            
            .actions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            button {
                padding: 15px 20px;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                color: white;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .action-0 { background: linear-gradient(135deg, #2196F3, #1976D2); }
            .action-1 { background: linear-gradient(135deg, #f44336, #D32F2F); }
            .action-2 { background: linear-gradient(135deg, #4CAF50, #388E3C); }
            .action-3 { background: linear-gradient(135deg, #FF9800, #F57C00); }
            .action-4 { background: linear-gradient(135deg, #9C27B0, #7B1FA2); }
            
            .reset-btn {
                background: linear-gradient(135deg, #607D8B, #455A64);
                width: 100%;
                margin-top: 20px;
            }
            
            .reward-section {
                background: linear-gradient(135deg, #FFD700, #FFA000);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }
            
            .reward-text {
                font-size: 24px;
                font-weight: bold;
                color: #fff;
            }
            
            .message {
                margin-top: 20px;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                display: none;
            }
            
            .message.success {
                background: #d4edda;
                color: #155724;
                display: block;
            }
            
            .message.error {
                background: #f8d7da;
                color: #721c24;
                display: block;
            }
            
            .status {
                margin-top: 20px;
                padding: 10px;
                text-align: center;
                color: #666;
                font-size: 12px;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .loading {
                animation: pulse 1s ease-in-out infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>
                <span>🧘</span>
                <span>Digital Detox Coach</span>
            </h1>
            <div class="subtitle">Your AI-powered assistant for healthy digital habits</div>
            
            <div class="metrics-grid" id="metrics">
                <div class="metric-card">
                    <div class="metric-title">📱 Screen Time</div>
                    <div class="metric-value" id="screen-time">0.0 h</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="screen-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">📚 Study Time</div>
                    <div class="metric-value" id="study-time">0.0 h</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="study-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">😴 Sleep Time</div>
                    <div class="metric-value" id="sleep-time">0.0 h</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="sleep-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">💪 Productivity</div>
                    <div class="metric-value" id="productivity">0%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="productivity-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">📱 Social Media Usage</div>
                    <div class="metric-value" id="social-time">0.0 h</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="social-progress" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            
            <div class="reward-section">
                <div class="reward-text" id="reward">🏆 Total Reward: 0</div>
            </div>
            
            <div class="actions-section">
                <h3>🎮 Take Action</h3>
                <div class="actions-grid">
                    <button class="action-0" onclick="takeAction(0)">📱 Allow App</button>
                    <button class="action-1" onclick="takeAction(1)">🚫 Block App</button>
                    <button class="action-2" onclick="takeAction(2)">📖 Suggest Study</button>
                    <button class="action-3" onclick="takeAction(3)">☕ Suggest Break</button>
                    <button class="action-4" onclick="takeAction(4)">😴 Suggest Sleep</button>
                </div>
                <button class="reset-btn" onclick="resetEnvironment()">🔄 Reset Environment</button>
            </div>
            
            <div id="message" class="message"></div>
            <div class="status" id="status">✅ System ready</div>
        </div>
        
        <script>
            let currentObservation = null;
            let totalReward = 0;
            
            async function resetEnvironment() {
                showMessage('Resetting environment...', 'info');
                setStatus('Resetting...');
                
                try {
                    const response = await fetch('/reset', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'}
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        currentObservation = data.observation;
                        totalReward = 0;
                        updateDisplay(currentObservation);
                        updateReward(0);
                        showMessage('✅ Environment reset successfully!', 'success');
                        setStatus('Ready');
                    } else {
                        showMessage('❌ Reset failed: ' + data.error, 'error');
                        setStatus('Error');
                    }
                } catch (error) {
                    showMessage('❌ Error: ' + error, 'error');
                    setStatus('Connection error');
                }
            }
            
            async function takeAction(action) {
                if (!currentObservation) {
                    await resetEnvironment();
                }
                
                const actionNames = ['Allow App', 'Block App', 'Suggest Study', 'Suggest Break', 'Suggest Sleep'];
                setStatus(`Taking action: ${actionNames[action]}...`);
                
                try {
                    const response = await fetch('/action', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({action: action})
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        currentObservation = data.next_state;
                        totalReward += data.reward;
                        updateDisplay(currentObservation);
                        updateReward(data.reward);
                        showMessage(`✅ ${actionNames[action]} - Reward: ${data.reward.toFixed(2)}`, 'success');
                        setStatus('Ready');
                        
                        if (data.done) {
                            showMessage('🏁 Episode complete! Click Reset to continue.', 'success');
                        }
                    } else {
                        showMessage('❌ Action failed: ' + data.error, 'error');
                        setStatus('Error');
                    }
                } catch (error) {
                    showMessage('❌ Error: ' + error, 'error');
                    setStatus('Connection error');
                }
            }
            
            function updateDisplay(obs) {
                // Update values
                document.getElementById('screen-time').textContent = obs[0].toFixed(1) + ' h';
                document.getElementById('study-time').textContent = obs[1].toFixed(1) + ' h';
                document.getElementById('sleep-time').textContent = obs[2].toFixed(1) + ' h';
                document.getElementById('productivity').textContent = Math.round(obs[3]) + '%';
                document.getElementById('social-time').textContent = obs[4].toFixed(1) + ' h';
                
                // Update progress bars
                updateProgress('screen-progress', obs[0], 24, obs[0].toFixed(1) + 'h');
                updateProgress('study-progress', obs[1], 12, obs[1].toFixed(1) + 'h');
                updateProgress('sleep-progress', obs[2], 12, obs[2].toFixed(1) + 'h');
                updateProgress('productivity-progress', obs[3], 100, Math.round(obs[3]) + '%');
                updateProgress('social-progress', obs[4], 24, obs[4].toFixed(1) + 'h');
            }
            
            function updateProgress(elementId, value, max, text) {
                const percent = (value / max) * 100;
                const element = document.getElementById(elementId);
                element.style.width = percent + '%';
                
                // Change color based on value
                if (elementId === 'screen-progress' || elementId === 'social-progress') {
                    if (value < max * 0.2) {
                        element.className = 'progress-fill good';
                    } else if (value < max * 0.5) {
                        element.className = 'progress-fill warning';
                    } else {
                        element.className = 'progress-fill danger';
                    }
                } else if (elementId === 'productivity-progress') {
                    if (value > 70) {
                        element.className = 'progress-fill good';
                    } else if (value > 40) {
                        element.className = 'progress-fill warning';
                    } else {
                        element.className = 'progress-fill danger';
                    }
                } else {
                    if (value > max * 0.6) {
                        element.className = 'progress-fill good';
                    } else if (value > max * 0.3) {
                        element.className = 'progress-fill warning';
                    } else {
                        element.className = 'progress-fill danger';
                    }
                }
            }
            
            function updateReward(reward) {
                const rewardElement = document.getElementById('reward');
                rewardElement.innerHTML = `🏆 Total Reward: ${totalReward.toFixed(2)}`;
                
                // Flash effect
                rewardElement.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    rewardElement.style.transform = 'scale(1)';
                }, 200);
            }
            
            function showMessage(msg, type) {
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = msg;
                messageDiv.className = 'message ' + type;
                setTimeout(() => {
                    messageDiv.className = 'message';
                }, 3000);
            }
            
            function setStatus(status) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = '🔄 ' + status;
            }
            
            // Initialize on page load
            resetEnvironment();
        </script>
    </body>
    </html>
    '''

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Main web interface"""
    return get_interface_html()

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "alive", "timestamp": os.popen('date').read().strip()}), 200

@app.route('/reset', methods=['POST', 'GET', 'OPTIONS'])
def reset():
    """Reset the environment"""
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        return response, 200
    
    try:
        observation = env.reset()
        
        if isinstance(observation, np.ndarray):
            observation = observation.tolist()
        
        print(f"✓ Reset successful - Observation: {observation}")
        
        return jsonify({
            "success": True,
            "observation": observation,
            "message": "Environment reset successful"
        }), 200
        
    except Exception as e:
        print(f"✗ Reset error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/action', methods=['POST', 'OPTIONS'])
def take_action():
    """Take an action in the environment"""
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    try:
        data = request.json
        action = data.get('action')
        
        if action is None:
            return jsonify({"error": "No action provided"}), 400
        
        if not isinstance(action, int) or action < 0 or action >= action_size:
            return jsonify({"error": f"Invalid action. Must be 0-{action_size-1}"}), 400
        
        print(f"Taking action: {action}")
        
        next_state, reward, done, info = env.step(action)
        
        if isinstance(next_state, np.ndarray):
            next_state = next_state.tolist()
        reward = float(reward)
        done = bool(done)
        
        print(f"✓ Action complete - Reward: {reward:.2f}, Done: {done}")
        
        return jsonify({
            "success": True,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "info": info
        }), 200
        
    except Exception as e:
        print(f"✗ Action error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/agent_action', methods=['POST', 'OPTIONS'])
def agent_action():
    """Get action from DQN agent"""
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
    
    try:
        data = request.json
        state = data.get('state')
        training = data.get('training', False)
        
        if state is None:
            return jsonify({"error": "No state provided"}), 400
        
        state = np.array(state)
        action = agent.act(state, training=training)
        
        return jsonify({
            "success": True,
            "action": action,
            "epsilon": agent.epsilon
        }), 200
        
    except Exception as e:
        print(f"✗ Agent action error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        "epsilon": agent.epsilon,
        "memory_size": len(agent.memory),
        "device": str(agent.device),
        "action_size": action_size,
        "state_size": state_size
    })

# ... (all your existing imports, routes, and other code above) ...

def main():
    """Entry point for digital-detox command"""
    print(f"🚀 Starting Flask app on port {PORT}")
    print(f"📍 Web Interface: http://0.0.0.0:{PORT}/")
    print(f"🔄 Reset endpoint: http://0.0.0.0:{PORT}/reset")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()
