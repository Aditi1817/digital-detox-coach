# 🧠 AI Digital Detox Coach

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/open-env)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

An intelligent digital wellness coach that uses **Reinforcement Learning** to help reduce screen time and promote healthy digital habits. Built with PyTorch and OpenEnv for the Meta PyTorch OpenEnv Hackathon.

## 🎯 Problem Statement

Digital addiction affects billions worldwide:
- 📱 Average screen time: 7+ hours daily
- 😴 Poor sleep quality due to late-night usage
- 📉 Decreased productivity and focus
- 🧠 Mental health impacts

Traditional solutions (timers, blockers) are static and don't adapt to individual behavior patterns.

## 💡 Our Solution

An **AI-powered Digital Detox Coach** that:
- 🧠 Learns from user behavior using Reinforcement Learning
- 🎯 Provides personalized suggestions (study, breaks, sleep)
- 📊 Tracks and visualizes progress
- 🔄 Adapts strategies based on effectiveness

## 🏗️ Architecture
┌─────────────────────────────────────────────────────┐
│ Digital Detox Coach │
├─────────────────────────────────────────────────────┤
│ Environment (OpenEnv) │ Agent (PyTorch) │
│ • Screen Time │ • DQN Network │
│ • Study Time │ • Experience Replay │
│ • Sleep Time │ • Target Network │
│ • Productivity │ • Epsilon-Greedy │
│ • Social Media │ │
├─────────────────────────────────────────────────────┤
│ Reward System (Optimized for Health) │
│ + Low screen time │ - High screen time │
│ + Study time │ - Social media abuse │
│ + Sleep quality │ - Late night usage │
└─────────────────────────────────────────────────────┘

