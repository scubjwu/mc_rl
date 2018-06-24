# -*- coding: utf-8 -*

from Env_modified import Env
from RL_brain_modified import DeepQNetwork


def run(RL):
    step = 0

    for episode in range(10000):
        print('episode = .................................', episode)
        with open('./result.txt', 'a') as res:
            res.write('episode = .................................' + str(episode) + '\n')

        # initial 
        init = True
        total_reward = 0
        env = Env()

        while True:

            if init is True:
                observation = env.reset()
                init = False

            # RL choose action based on observation
            action = RL.choose_action(observation, env.get_evn_time())

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            #print 'reward: ', reward
            total_reward += reward

            RL.store_transition(observation, action, reward, env.get_evn_time(), observation_)

            #if (step > 5000) and (step % 25 == 0):
            if (step > 500) and (step % 5 == 0):
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
    RL = DeepQNetwork()

    run(RL)
    #RL.plot_cost()
