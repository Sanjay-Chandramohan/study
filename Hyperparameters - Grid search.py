import gym
import numpy as np
import random
import csv
import itertools
import matplotlib.pyplot as plt

# Loading and rendering the gym environment

env = gym.envs.make("CliffWalking-v0")
env.reset()

epsilon = 1
max_epsilon = 1
min_epsilon = 0.1
train_episodes = 500
test_episodes = 100
max_steps = 100
log_interval = 10

alpha_lists = np.linspace(0.0, 1.0, num=11)
alpha_lists = [round(value, 2) for value in alpha_lists]
discount_factor_lists = np.linspace(0.0, 1.0, num=11)
discount_factor_lists = [round(value, 2) for value in discount_factor_lists]
decay_lists = np.linspace(0.001, 0.1, num=10)
decay_lists = [round(value, 2) for value in decay_lists]
# print(discount_factor_lists)


combo = list(itertools.product(alpha_lists, discount_factor_lists, decay_lists))
# print(len(combo))
iteration = 0
for i in range(len(combo)):
    print("combo" + str(i), combo[i])
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilons = []
    training_rewards = []
    alpha, discount_factor, decay = combo[i]
    for episode in range(train_episodes):
        # print(episode)
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # print(step)
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            # STEP 2: SECOND option for choosing the initial action - exploit
            # If the random number is larger than epsilon: employing exploitation
            # and selecting best action
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

            # STEP 2: FIRST option for choosing the initial action - explore
            # Otherwise, employing exploration: choosing a random action
            else:
                action = env.action_space.sample()

            # STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)

            # STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done == True:
                # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                break
        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    # print("All training episode rewards:")
    # print(training_rewards)
    # print("Maximum score:" + str(max(training_rewards)))
    # print("Minimum score:" + str(min(training_rewards)))

    print("Training score over time: " + str(sum(training_rewards) / train_episodes))
    start = 0
    avg_reward = []
    cycles = train_episodes / log_interval
    cycles = int(cycles)
    # print(cycles)
    # print("average score per " + str(log_interval) + " episodes")
    for epoch in range(cycles):
        # print("epoch:", epoch + 1)
        # print("episode:", start)
        add = sum(training_rewards[start:start + (log_interval - 1)]) / log_interval
        # print(add)
        avg_reward.append(add)
        start = start + log_interval
        # print("start:", start)

    # print(avg_reward)
    # print("Plotting rewards:")
    axes = np.linspace(1, (train_episodes / log_interval), num=int(train_episodes / log_interval))
    # print(axes)
    plt.xlim(0, 25)
    plt.ylim(-1000, 0)
    plt.plot(axes, avg_reward)
    plt.xlabel('epochs(10 episodes)')
    plt.ylabel("average score per epoch")
    # plt.plot(training_rewards)

    path = "C:/Users/sanja/Desktop/Thesis/experiments/exp4/CliffWalking-v0_figs/" + "combo" + str(iteration) + ".png"
    plt.savefig(path)
    # plt.show()

    '''addrew = str(avg_reward)
    comb = str(combo)
    with open('C:/Users/sanja/Desktop/Thesis/experiments/exp4/CliffWalking-v0_result.txt', 'a') as f:
        f.write(comb)
        f.write(addrew)
        f.write('\n')
        f.close()
    iteration += 1'''

    with open('C:/Users/sanja/Desktop/Thesis/experiments/exp4/CliffWalking-v0_hyper-parameter.csv', 'ab') as csvfile:
        # csvwriter = csv.writer(csvfile, delimiter=';')
        hyper = [alpha, discount_factor, decay]
        arr1 = np.array(hyper)
        np.savetxt(csvfile, arr1, fmt='%-5.4f', delimiter=',')

    with open('C:/Users/sanja/Desktop/Thesis/experiments/exp4/CliffWalking-v0_results.csv', 'ab') as csvfile:
        # csvwriter = csv.writer(csvfile, delimiter=';')
        res = training_rewards
        arr2 = np.array(res)
        np.savetxt(csvfile, arr2, fmt='%-5.4f', delimiter=',')

    iteration += 1
