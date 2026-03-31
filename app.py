"""
Hugging Face Spaces App for Digital Detox Coach
Interactive web interface for the AI Digital Detox Coach
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from environment import DigitalDetoxEnv
from agent import DQNAgent
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Digital Detox Coach",
    page_icon="🧘",
    layout="wide"
)

# Title and description
st.title("🧘 AI Digital Detox Coach")
st.markdown("""
    ### Learn healthy digital habits with Reinforcement Learning!
    
    This AI coach uses Deep Q-Learning to suggest optimal digital behaviors 
    that reduce screen time and improve productivity.
""")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Model selection
model_path = st.sidebar.selectbox(
    "Select Model",
    ["results/final_detox_agent.pth", "results/detox_agent_episode_500.pth", "Untrained"]
)

# Simulation parameters
episodes = st.sidebar.slider("Number of Episodes", 1, 10, 3)
max_steps = st.sidebar.slider("Hours per Day", 12, 48, 24)

# Initialize environment and agent
@st.cache_resource
def load_agent(model_path):
    agent = DQNAgent(state_size=5, action_size=5)
    if model_path != "Untrained":
        try:
            agent.load(model_path)
        except:
            st.warning(f"Model not found at {model_path}, using untrained agent")
    return agent

agent = load_agent(model_path)

# Create environment
env = DigitalDetoxEnv({'max_steps': max_steps})

# Action names for display
action_names = {
    0: "📱 Allow App",
    1: "🚫 Block App",
    2: "📚 Suggest Study",
    3: "☕ Suggest Break",
    4: "😴 Suggest Sleep"
}

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    st.session_state['running'] = True
    st.session_state['results'] = None

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Simulation", "📈 Analysis", "🤖 Agent Info", "📖 About"])

with tab1:
    if 'running' in st.session_state and st.session_state['running']:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store results
        all_rewards = []
        all_actions = []
        all_states = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_actions = []
            episode_states = []
            
            status_text.text(f"Running Episode {episode + 1}/{episodes}...")
            
            for step in range(max_steps):
                # Choose action
                action = agent.act(state, training=False)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store data
                episode_actions.append(action)
                episode_states.append(state.copy())
                
                state = next_state
                total_reward += reward
                
                # Update progress
                progress = (episode * max_steps + step + 1) / (episodes * max_steps)
                progress_bar.progress(progress)
            
            all_rewards.append(total_reward)
            all_actions.append(episode_actions)
            all_states.append(episode_states)
            
            # Display episode summary
            with st.expander(f"Episode {episode + 1} Summary"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Screen Time", f"{state[0]:.1f}h", 
                             delta="Healthy" if state[0] < 5 else "High")
                with col2:
                    st.metric("Study Time", f"{state[1]:.1f}h",
                             delta="Good" if state[1] > 4 else "Low")
                with col3:
                    st.metric("Sleep", f"{state[2]:.1f}h",
                             delta="Optimal" if 7 <= state[2] <= 9 else "Suboptimal")
                with col4:
                    st.metric("Productivity", f"{state[3]:.1f}%",
                             delta="High" if state[3] > 80 else "Low")
                with col5:
                    st.metric("Total Reward", f"{total_reward:.1f}")
        
        progress_bar.progress(1.0)
        status_text.text("Simulation Complete!")
        
        # Store results in session
        st.session_state['results'] = {
            'rewards': all_rewards,
            'actions': all_actions,
            'states': all_states
        }
        
        # Show summary metrics
        st.subheader("📊 Overall Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Reward", f"{np.mean(all_rewards):.1f}")
        with col2:
            st.metric("Best Episode", f"{np.max(all_rewards):.1f}")
        with col3:
            st.metric("Success Rate", 
                     f"{(np.array(all_rewards) > 0).sum() / len(all_rewards) * 100:.1f}%")

with tab2:
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        # Reward plot
        st.subheader("📈 Episode Rewards")
        fig_rewards = go.Figure()
        fig_rewards.add_trace(go.Scatter(
            y=results['rewards'],
            mode='lines+markers',
            name='Rewards',
            line=dict(color='blue', width=2)
        ))
        fig_rewards.update_layout(
            xaxis_title="Episode",
            yaxis_title="Total Reward",
            hovermode='x'
        )
        st.plotly_chart(fig_rewards, use_container_width=True)
        
        # State evolution for last episode
        if results['states']:
            last_episode_states = np.array(results['states'][-1])
            
            st.subheader("📊 Last Episode State Evolution")
            
            # Create DataFrame for plotting
            df = pd.DataFrame(
                last_episode_states,
                columns=['Screen Time', 'Study Time', 'Sleep Time', 
                        'Productivity', 'Social Media']
            )
            df['Hour'] = range(len(df))
            
            # Plot state evolution
            fig = go.Figure()
            for col in df.columns[:-1]:  # Exclude Hour
                fig.add_trace(go.Scatter(
                    x=df['Hour'],
                    y=df[col],
                    mode='lines',
                    name=col
                ))
            
            fig.update_layout(
                xaxis_title="Hour",
                yaxis_title="Value",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Action distribution
            if results['actions']:
                st.subheader("🎯 Action Distribution")
                last_episode_actions = results['actions'][-1]
                action_counts = pd.Series(last_episode_actions).value_counts()
                
                fig_actions = go.Figure(data=[go.Pie(
                    labels=[action_names[i] for i in action_counts.index],
                    values=action_counts.values,
                    hole=.3
                )])
                fig_actions.update_layout(title="Actions Taken")
                st.plotly_chart(fig_actions, use_container_width=True)
    else:
        st.info("Run a simulation first to see analysis!")

with tab3:
    st.header("🤖 Agent Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent Architecture")
        st.code("""
        DQN Agent:
        - State Size: 5 (Screen Time, Study Time, Sleep Time, 
                         Productivity, Social Media)
        - Action Space: 5 (Allow, Block, Study, Break, Sleep)
        - Neural Network: 3 hidden layers (128 units each)
        - Replay Buffer: 10,000 experiences
        - Gamma: 0.99
        - Epsilon Decay: 0.995
        """)
    
    with col2:
        st.subheader("Reward Logic")
        st.markdown("""
        **Positive Rewards:**
        - ✅ Low screen time (<4h): +10
        - ✅ High study time (>4h): +15
        - ✅ Optimal sleep (7-9h): +20
        - ✅ Taking breaks: +5
        - ✅ Blocking apps: +10
        
        **Negative Rewards:**
        - ❌ High screen time (>8h): -15
        - ❌ Excessive social media (>3h): -20
        - ❌ Late night usage: -25
        - ❌ Low productivity: -10
        """)

with tab4:
    st.header("📖 About AI Digital Detox Coach")
    
    st.markdown("""
    ### Problem Statement
    Digital addiction is a growing concern affecting productivity, mental health, 
    and quality of life. People spend an average of 7+ hours on screens daily, 
    leading to decreased focus, sleep issues, and reduced well-being.
    
    ### Solution
    This AI Digital Detox Coach uses **Reinforcement Learning** to learn optimal 
    digital behavior patterns. It suggests actions that:
    - 📱 Manage app usage intelligently
    - 📚 Encourage productive activities
    - 😴 Promote healthy sleep habits
    - ☕ Suggest timely breaks
    
    ### How It Works
    1. **Environment**: Simulates a digital day with screen time, study, sleep, 
       productivity, and social media metrics
    2. **Agent**: Uses Deep Q-Learning to learn optimal policies
    3. **Rewards**: Designed to encourage healthy digital habits
    4. **Training**: 500+ episodes to learn effective strategies
    
    ### Technology Stack
    - **PyTorch**: Deep Learning framework
    - **OpenEnv**: Reinforcement Learning environment
    - **Streamlit**: Interactive web interface
    - **Plotly**: Interactive visualizations
    
    ### Benefits
    - 🎯 Personalized digital behavior suggestions
    - 📊 Real-time monitoring and feedback
    - 🔬 Data-driven insights
    - 🧠 Learn from past behaviors
    
    ### Future Improvements
    - Personal user profiles
    - Real app integration
    - Mobile app version
    - Multi-user comparisons
    - Advanced reward shaping
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using PyTorch, OpenEnv, and Streamlit | Meta PyTorch OpenEnv Hackathon")