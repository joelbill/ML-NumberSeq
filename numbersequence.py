import numpy as np
import random

NUMBERS = 20
ACTIONS = 2
qtable = np.zeros((NUMBERS*NUMBERS, ACTIONS))
total_episodes = 50        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 20000                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob

# List of rewards
rewards = []


def act(action, state, index): # take the action. 0 = DECREASE, 1 = INCREASE

    rward = 0
    if state == index + 1:
        rward = 1
        return state, rward
    else:
        if action == 0:
            if state == 0:
                newstate = state
            else:
                newstate = updated_numbers[index] - 1
        else:
            if state == 20:
                newstate = state
            else:
                newstate = updated_numbers[index] + 1

        if newstate == index + 1:
            rward = 1

        return newstate, rward



for episode in range(total_episodes):
    # List of input numbers
    input_numbers = []
    for i in range(0, NUMBERS):
        input_numbers.append(random.randint(1,NUMBERS))

    # Reset the environment
    updated_numbers = input_numbers
    index = 0
    state = 0
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):

        if index > NUMBERS - 1:
            break

        exploit_explore_rand = random.uniform(0, 1)

        # if randomed number is greater than epsilon --> exploit from q-table
        if exploit_explore_rand > epsilon:
            action = np.argmax(qtable[state, :])

        # else take random action --> exploration
        else:
            action = random.randint(0, 1)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward = act(action, state, index)
        if new_state == 2 and index == 0:
            # print('2')
            h = 1
        if new_state == 3 and index == 1:
            # print('3')
            h = 2
        if new_state == 4 and index == 2:
            # print('4')
            h = 3

        if index == 18 and index == 15:
            h = 5

        updated_numbers[index] = new_state
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        if index*NUMBERS + state - 1 == 399:
            h = 6

        qtable[index*NUMBERS + state - 1, action] = qtable[index*NUMBERS + state - 1, action] + learning_rate * (
                reward + gamma * np.max(qtable[index*NUMBERS + new_state - 1, :]) - qtable[index*NUMBERS + state - 1, action])

        total_rewards = total_rewards + reward

        # Our new state is state
        state = new_state

        # If reward = 1 we have found the right number at that index
        if reward == 1:
            index += 1

    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print(qtable)
