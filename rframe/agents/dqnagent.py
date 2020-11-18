import numpy
import collections

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow
import random
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, env, replay_size, merge_freq, save_dir):
        self.env = env
        self.replay_size = replay_size
        self.merge_freq = merge_freq
        self.save_dir = save_dir

        self.replay = collections.deque(maxlen=replay_size)
        
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.merge_cntr = 0

    def _create_model(self):
        model = tensorflow.keras.models.Sequential()
        try:
            model.load(self.save_dir)
        except:
            model.add(tensorflow.keras.layers.Dense(self.env.obs_s * 10, input_shape=(self.env.obs_s,), activation="sigmoid"))
            model.add(tensorflow.keras.layers.Dense(self.env.obs_s * 10, activation="sigmoid"))
            model.add(tensorflow.keras.layers.Dense(self.env.act_s, activation="linear"))
            model.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def fit(self, episodes, batch_size, gamma, epsilon, epsilon_min, epsilon_decay):
        statistics = [[], []]

        for episode in range(episodes):
            state, reward, done = self.env.reset(), 0, False

            total_reward = 0
            while not done:
                if numpy.random.random() > epsilon:
                    action = numpy.argmax(self._predict(state))
                else:
                    action = numpy.random.randint(0, self.env.act_s)

                new_state, reward, done = self.env.step(action)
                total_reward += reward

                if episode % 100 == 0:
                    self.env.render()

                self.replay.append([state, action, new_state, reward, done])
                self._train(batch_size, gamma, done)

                state = new_state
                print(f"Episode: {episode} | Epsilon: {round(epsilon, 3)} | Reward: {total_reward}", end='\r')
            print()
            self._save()

            if epsilon >= epsilon_min:
                epsilon *= epsilon_decay

            statistics[0].append(episode)
            statistics[1].append(total_reward)

        plt.plot(*statistics)
        plt.show()

    def _train(self, batch_size, gamma, terminated):
        if len(self.replay) < self.replay_size:
            return

        batch = random.sample(self.replay, batch_size)

        um_states = numpy.array([item[0] for item in batch])
        um_states_qs = numpy.array(self.model(um_states))

        new_um_states = numpy.array([item[2] for item in batch])
        new_um_states_qs = numpy.array(self.target_model(new_um_states))

        X, Y = [], []
        for i, (state, action, new_state, reward, done) in enumerate(batch):
            if not done:
                max_future_q = numpy.max(new_um_states_qs[i])
                new_q = reward + gamma * max_future_q
            else:
                new_q = reward

            m_qs = um_states_qs[i]
            m_qs[action] = new_q

            X.append(state)
            Y.append(m_qs)

        self.model.train_on_batch(numpy.array(X), y=numpy.array(Y)) # , batch_size=batch_size, verbose=0, shuffle=False)

        if terminated:
            self.merge_cntr += 1
        if self.merge_cntr > self.merge_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.merge_cntr = 0

    def _predict(self, state):
        numpy_state = numpy.array(state)
        return numpy.array(self.model(numpy_state.reshape(-1, *numpy_state.shape))[0])

    def _save(self):
        self.model.save(self.save_dir)
