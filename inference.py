from environment import DigitalDetoxEnv

env = DigitalDetoxEnv()

def reset():
    state = env.reset()
    return {
        "state": state.tolist()
    }

def step(action):
    state, reward, done, info = env.step(action)

    return {
        "state": state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }