import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered

# LEARNING_RATE = 0.1
LEARNING_RATE = [0.1, 0.4, 0.66]

# Between 0 and 1, represents of how much we care about future reward over immediate reward
DISCOUNT = 0.95
# DISCOUNT = [0.95, 0.75 , 0.5]

RUNS = 10000  # Number of  total iterations

UPDATE_EVERY = 100  # How often the current progress is recorded

previousCount = []  # array of all scores over runs
numofruns = []
avgscore = []
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics for graph

# for j in DISCOUNT:
for k in LEARNING_RATE:
    def create_bins():
        numbins = 50
        bins = [np.linspace(-4.8, 4.8, numbins),
                np.linspace(-4, 4, numbins),
                np.linspace(-0.418, 0.418, numbins),
                np.linspace(-4, 4, numbins)
                ]
        return bins, numbins


    def create_qtable(numofbins):
        obsSpace = len(env.observation_space.high)
        qTable = np.zeros([numofbins] * obsSpace + [env.action_space.n])
        return qTable, obsSpace


    def get_discrete_state(state, binsize, obs):
        stateIndex = []
        for i in range(obs):
            digital = np.digitize(state[i], binsize[i]) - 1
            # digital = binsize[i][digital]
            stateIndex.append(digital)
        return tuple(stateIndex)


    '''def get_discrete_state(state, bins, obsSpaceSize):
        stateIndex = []
        for i in range(obsSpaceSize):
            stateIndex.append(np.digitize(state[i], bins[i]) - 1)  # -1 will turn bin into index
        return tuple(stateIndex)'''

    bins, numbins = create_bins()
    qTable, obsSpace = create_qtable(numbins)
    epsilon = 1
    START_EPSILON_DECAYING = 1
    # Change Epsilon decay duration below
    END_EPSILON_DECAYING = RUNS // 2
#   END_EPSILON_DECAYING = (7.5 * RUNS) // 10
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    for run in range(RUNS):
        discreteState = get_discrete_state(env.reset(), bins, obsSpace)
        done = False  # has the environment finished?
        cnt = 0  # how may movements cart has made

        while not done:

            cnt += 1
            # Get action from Q table
            if np.random.random() > epsilon:
                action = np.argmax(qTable[discreteState])
            # Get random action
            else:
                action = np.random.randint(0, env.action_space.n)
            newState, reward, done, _ = env.step(action)  # perform action on environment

            newDiscreteState = get_discrete_state(newState, bins, obsSpace)

            maxFutureQ = np.max(qTable[newDiscreteState])  # estimate optimal future value

            oldQ = qTable[discreteState + (action,)]  # old Q-value

            # if pole fell over or went out of bounds, negative reward
            if done and cnt < 200:
                reward = -375

            # formula to calculate all new Q values
            newQ = (1 - k) * oldQ + k * (reward + DISCOUNT * maxFutureQ)
            #           newQ = (1 - LEARNING_RATE) * oldQ + LEARNING_RATE * (reward + j * maxFutureQ)
            qTable[discreteState + (action,)] = newQ  # Update qTable with new Q value

            discreteState = newDiscreteState

        previousCount.append(cnt)

        # Decaying is being done every run if run number is within decaying range
        if END_EPSILON_DECAYING >= run >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        # Add new metrics for graph
        update = run + 1
        if update % UPDATE_EVERY == 0:
            latestRuns = previousCount[-UPDATE_EVERY:]
            averageCnt = sum(latestRuns) / len(latestRuns)
            numofruns.append(update)
            avgscore.append(averageCnt)
            metrics['ep'].append(update)
            metrics['avg'].append(averageCnt)
            metrics['min'].append(min(latestRuns))
            metrics['max'].append(max(latestRuns))
            print("Run:", update, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

env.close()

print("Plotting average rewards:")
lengthofnum = len(numofruns)
lengthofavg = len(avgscore)
num1 = numofruns[:int(lengthofnum / 3)]
num2 = numofruns[int(lengthofnum / 3):int(2 * lengthofnum / 3)]
num3 = numofruns[int(2 * lengthofnum / 3):]
avg1 = avgscore[:int(lengthofavg / 3)]
avg2 = avgscore[int(lengthofavg / 3):int(2 * lengthofavg / 3)]
avg3 = avgscore[int(2 * lengthofavg / 3):]
print("avg1", avg1)
print("avg2", avg2)
print("avg3", avg3)

plt.title("Q-learning average scores with different learning rates")
plt.plot(numofruns[:int(lengthofnum / 3)], avgscore[:int(lengthofavg / 3)],
         label="Learning rate = " + str(LEARNING_RATE[0]))
plt.plot(numofruns[int(lengthofnum / 3):int(2 * lengthofnum / 3)],
         avgscore[int(lengthofavg / 3):int(2 * lengthofavg / 3):],
         label="Learning rate = " + str(LEARNING_RATE[1]))
plt.plot(numofruns[int(2 * lengthofnum / 3):], avgscore[int(2 * lengthofavg / 3):],
         label="Learning rate = " + str(LEARNING_RATE[2]))
plt.xlabel('number of runs')
plt.ylabel('average score')
plt.legend()
plt.show()

'''plt.title("Q-learning average scores with different discount rates")
plt.plot(numofruns[:int(lengthofnum / 3)], avgscore[:int(lengthofavg / 3)],
         label="Discount rate = " + str(DISCOUNT[0]))
plt.plot(numofruns[int(lengthofnum / 3):int(2 * lengthofnum / 3)],
         avgscore[int(lengthofavg / 3):int(2 * lengthofavg / 3):],
         label="Discount rate = " + str(DISCOUNT[1]))
plt.plot(numofruns[int(2 * lengthofnum / 3):], avgscore[int(2 * lengthofavg / 3):],
         label="Discount rate = " + str(DISCOUNT[2]))
plt.xlabel('number of runs')
plt.ylabel('average score')
plt.legend()
plt.show()'''
