from core import dnq
import numpy as np

start = time.time()
aggregated_agents = []
time_data = []
for rep in [[0.025, 10**(-6)], [0.1, 0.5*10**(-5)], [0.2, 10**(-5)], [0.05, 1.5*10**(-5)], [0.2, 10**(-6)]]:
    total_rewards = []

    for loop in range(20):
        print("Loop:", loop, "alpha:", rep[0], "beta:", rep[1])
        agents = [dnq.Agent(alpha=rep[0], beta=rep[1], doubleQ=True)
                  for _ in range(2)]
        env = dnq.Env()

        temps = []
        rewards = []
        epsilon = []
        prices = []

        # Initialization of prices p0 (done directly in each agent)
        for agent in agents:
            agent.p = np.random.choice(agent.A)

        # Initialization of state
        s_t = env([agent.p for agent in agents])[1]
        for agent in agents:
            agent.s_t = s_t

        s_ind = agents[0].find_index(agents[0].s_t)
        for agent in agents:
            agent.s_ind = s_ind

        # Iterative phase
        for t in range(10**6):
            if t % (2*10**5) == 0:
                inter_start = time.time()
                print("t:", t)

            # Actions and state at t+1
            for agent in agents:
                action = agent.choose_action(observation)
                ret = env(s_t1)
                quant, price, cost = ret

                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward,
                                       observation_, done)
                agent.learn()
                observation = observation_

                agent.a_ind = agent.get_next_action()

            s_t1 = env([agent.A[agent.a_ind] for agent in agents])[1]
            for agent in agents:
                agent.s_t1 = s_t1

            s_ind1 = agents[0].find_index(agents[0].s_t1)
            for agent in agents:
                agent.s_ind1 = s_ind1

            temps.append(t)
            ret = env(s_t1)
            quant, price, cost = ret

            re = ret[0]*ret[1]-ret[0]*ret[2]
            rewards.append(re)
            epsilon_values = [agent.epsilon for agent in agents]
            epsilon.append(epsilon_values)
            prices.append([agent.p for agent in agents])

            for i, agent in enumerate(agents):
                agent.updateQ(q=quant[i], p=price[i], c=cost[i], t=t)

            if t % (2*10**5) == 0:
                inter_end = time.time()
                time_data.append(inter_end-inter_start)
                print('average CPU', np.mean(time_data))

        total_rewards.append(rewards)

    aggregated_agents.append(np.array(total_rewards))

end = time.time()

with open('data_dQ.pkl', 'wb') as f:
    pickle.dump(aggregated_agents, f)

print(aggregated_agents)
print(end-start)

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
