from core import dnq
import numpy as np

start = time.time()
aggregated_agents = []
time_data = []
for rep in [[0.025, 10**(-6)], [0.1, 0.5*10**(-5)], [0.2, 10**(-5)], [0.05, 1.5*10**(-5)], [0.2, 10**(-6)]]:
    total_rewards = []

    for loop in range(20):
        print("Loop:", loop, "alpha:", rep[0], "beta:", rep[1])
        agents = [dnq.Agent(alpha=rep[0], beta=rep[1])
                  for _ in range(2)]
        env = dnq.Env()

        temps = []
        rewards = []
        epsilon = []
        prices = []

        # Initialization of prices p0 (done directly in each agent)
        state = []
        for agent in agents:
            action = np.random.choice(agent.A)
            state.append(action)

        reward = []
        quant, price, cost = env(state)
        re = quant*price - quant*cost
        reward.append(re)

        for i, agent in enumerate(agents):
            agent.transition(state, action, reward, state_1)

        for t in range(10):

            state_1 = []
            for agent in agents:
                action = agent.get_next_action(state)
                state_1.append(action)


if __name__ == '__main__':
    env = dnq.Env()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)
