from evaluation import evaluate
from ppo_agent import PPOAgent
from car_racing_v2 import CarRacingV2

if __name__ == '__main__':
    # Create environment
    env = CarRacingV2()

    # Create and train agent
    agent = PPOAgent(env)
    agent.train(episodes=2000, batch_size=512)

    # Evaluate agent
    env = CarRacingV2(step_repeat=1)
    evaluate(env, agent, 'car.mp4')
