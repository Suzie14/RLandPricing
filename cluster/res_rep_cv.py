import numpy as np
import pickle
import time

from core import qlearning as q


start = time.time()
final_rewards_all = []
mean_lc = []
time_data = []
for rep in [[0.025, 10**(-6)], [0.1, 0.5*10**(-5)], [0.2, 10**(-5)], [0.05, 1.5*10**(-5)], [0.2, 10**(-6)]]:
    final_rewards = []
    store_rewards_lc = []

    for loop in range(20):
        print("Loop:", loop, "alpha:", rep[0], "beta:", rep[1])
        agents = [q.Agent(alpha=rep[0], beta=rep[1]) for _ in range(2)]
        env = q.Env()

        temps = []
        last_100_values = [[None, None]] * 100
        rewards_lc = []
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

        t = 0
        done_forall = False
        # Iterative phase
        while not done_forall:

            if t >= 5*(10**6):
                break

            if t % (2*(10**5)) == 0:
                inter_start = time.time()
                print("t:", t)

            # Actions and state at t+1
            nb_done = 0
            for agent in agents:
                agent.a_ind = agent.get_next_action()
                # check here if the same action was taken in the past for the same state:
                agent.final_state()
                if agent.done == True:
                    nb_done += 1
                if nb_done == len(agents):
                    done_forall = True

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

            last_100_values.append(re)
            last_100_values.pop(0)
            if t % 100 == 0:
                rewards_lc.append(re)
            epsilon_values = [agent.epsilon for agent in agents]
            epsilon.append(epsilon_values)
            prices.append([agent.p for agent in agents])

            for i, agent in enumerate(agents):
                agent.updateQ(q=quant[i], p=price[i], c=cost[i], t=t)

            if t % (2*(10**5)) == 0:
                inter_end = time.time()
                time_data.append(inter_end-inter_start)
                print('average CPU', np.mean(time_data))

            t += 1
        rewards_lc = np.array(rewards_lc)
        max_length_rlc = int(5*(10**6)/100)
        if len(rewards_lc) < max_length_rlc:
            rewards_lc = np.vstack(rewards_lc)
            pad = np.tile(rewards_lc[-1], (max_length_rlc-len(rewards_lc), 1))
            rewards_lc = np.vstack((rewards_lc, pad))
        store_rewards_lc.append(rewards_lc)
        final_rewards.append(last_100_values)
        print('DONE, sample = ', loop+1, ', duration = ', t, ' periods')
    store_rewards_lc = np.array(store_rewards_lc)
    mean_lc.append(store_rewards_lc.mean(axis=0))
    final_rewards_all.append(final_rewards)

end = time.time()

with open('lc_rep_cv.pkl', 'wb') as f:
    pickle.dump(mean_lc, f)

with open('final_rep_cv.pkl', 'wb') as f:
    pickle.dump(final_rewards_all, f)

print(final_rewards_all)
print(end-start)
