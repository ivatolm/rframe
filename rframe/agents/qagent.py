import numpy
import matplotlib.pyplot as plt


class QAgent:
    def __init__(self, env):
        self.env = env
        self.qtable = {}

    def fit(self, episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        statistics = [[], []]

        for episode in range(episodes):
            print(f"Episode: {episode} | Epsilon: {round(epsilon, 3)}")

            state, reward, done = self.env.reset(), 0, False
            state = self._cnv_st(state)
            self._crnex_st(state)

            total_reward = 0
            while not done:
                if numpy.random.random() > epsilon:
                    action = numpy.argmax(self.qtable[state])
                else:
                    action = numpy.random.randint(0, self.env.act_s)

                new_state, reward, done = self.env.step(action)
                new_state = self._cnv_st(new_state)
                self._crnex_st(new_state)

                if not done:
                    max_future_q = numpy.max(self.qtable[new_state])
                    current_q = self.qtable[state][action]
                    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
                    self.qtable[state][action] = new_q

                state = new_state

                total_reward += reward
            statistics[0].append(episode)
            statistics[1].append(total_reward)

            if epsilon >= epsilon_min:
                epsilon -= epsilon_decay
        
        plt.plot(*statistics)
        plt.show()

    def _cnv_st(self, state):
        return str(state)

    def _crnex_st(self, state):
        if state not in self.qtable:
            self.qtable[state] = [numpy.random.random()] * self.env.act_s