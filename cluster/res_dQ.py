import numpy as np
import pickle
import time

from core import qlearning as q


start = time.time()
aggregated_agents = []
time_data = []
for rep in [[0.025, 10**(-6)], [0.1, 0.5*10**(-5)], [0.2, 10**(-5)], [0.05, 1.5*10**(-5)], [0.2, 10**(-6)]]:
    total_rewards = []

    for loop in range(20):
        print("Loop:", loop, "alpha:", rep[0], "beta:", rep[1])
        agents = [q.Agent(alpha=rep[0], beta=rep[1], doubleQ=True)
                  for _ in range(2)]
        env = q.Env()

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

    aggregated_agents.append(np.array(total_rewards).mean(axis=0))

end = time.time()

with open('data_dQ.pkl', 'wb') as f:
    pickle.dump(aggregated_agents, f)

print(aggregated_agents)
print(end-start)
