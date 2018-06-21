# -*- coding: utf-8 -*

from Env_modified import Env
from RL_brain_modified import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(4000):
        print('episode = .................................', episode)
        with open('./result.txt', 'a') as res:
            res.write('episode = .................................' + str(episode) + '\n')

        # initial 
        init = True
        total_reward = 0
        env = Env()

        while True:

            if init is True:
                observation, reward, done = env.reset(RL)
                init = False

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            total_reward += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print('total step:', step)
                print('total reward         ', total_reward)

                with open('./result.txt', 'a') as res:
                    res.write('总奖励:    ' + str(total_reward) + '\n')

                break

            step += 1

    # end of game
    print('game over')


if __name__ == "__main__":
    # game
    RL = DeepQNetwork(learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=3000,
                      # output_graph=True
                      )
    run_maze()
    #RL.plot_cost()
