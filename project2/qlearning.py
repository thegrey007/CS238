import numpy as np
import pandas as pd
import torch.nn as nn
import argparse
import bisect
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from time import time

def state_to_indices(position, velocity):
    pos_idx = int((position + 1.2) / (0.6 + 1.2) * 499)  # scale to [0, 499]
    vel_idx = int((velocity + 0.07) / (0.07 + 0.07) * 99)  # scale to [0, 99]
    return pos_idx, vel_idx

def indices_to_state(pos_idx, vel_idx):
    position = pos_idx / 499 * (0.6 + 1.2) - 1.2
    velocity = vel_idx / 99 * (0.07 + 0.07) - 0.07
    return position, velocity

def state_to_discretized_vals(state):
    pos_idx = (state - 1) % 500
    vel_idx = (state - 1) // 500
    return (pos_idx, vel_idx)

def nearest_visited_state(visited_list, state):
    # binary search for nearest visited state
    idx = bisect.bisect_left(visited_list, state)
    if idx == 0:
        nearest = visited_list[0]
    elif idx == len(visited_list):
        nearest = visited_list[-1]
    else:
        left = visited_list[idx - 1]
        right = visited_list[idx]
        nearest = left if abs(left - state) <= abs(right - state) else right
    return nearest

def nearest_visited_state_mountaincar(visited_list, s):
    pos_s , vel_s = state_to_discretized_vals(s + 1)
    best_dist = float('inf')
    nearest = None
    for v in visited_list:
        pos_v , vel_v = state_to_discretized_vals(v + 1)
        dist = abs(pos_v - pos_s) + abs(vel_v - vel_s)
        if dist < best_dist:
            best_dist = dist
            nearest = v
    return nearest

def fitted_q_iteration_poly(data, n_states, n_actions, gamma=0.99, n_iters=50, degree=5):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
    q_values = np.zeros((n_states, n_actions))

    for it in range(n_iters):
        X, y = [], []
        for _, row in data.iterrows():
            s, a, r, sp = row['s'] - 1, row['a'] - 1, row['r'], row['sp'] - 1
            target = r + gamma * np.max(q_values[sp])
            X.append([s / n_states, a / n_actions]) # to keep range same
            y.append(target)

        X, y = np.array(X), np.array(y)
        model.fit(X, y)

        sa_pairs = np.array([[s/n_states, a/n_actions] for s in range(n_states) for a in range(n_actions)])
        q_pred = model.predict(sa_pairs)
        q_values = q_pred.reshape(n_states, n_actions)
        print(f"Iteration {it+1}/{n_iters} done.")

    return np.argmax(q_values, axis=1) + 1

def q_learning(task_name, data, n_states, n_actions, alpha=0.1, gamma=0.95, epochs=10):
    # Initialize Q-table
    q_table = np.zeros((n_states, n_actions))  # n_states and n_actions
    # maintain a set of visited states
    visited_states = np.zeros(n_states, dtype=bool)
    visited_list = []  # keep sorted list of visited states
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for _, row in data.sample(frac=1, random_state=epoch).iterrows():
            s = row['s'] - 1 # since states are 1-indexed in the data
            a = row['a'] - 1 # since actions are 1-indexed in the data
            r = row['r']
            s_next = row['sp'] - 1 # since states are 1-indexed in the data
            # for q_table, find nearest state that we have visited
            # mark visited
            if not visited_states[s]:
                bisect.insort(visited_list, s)  # insert while keeping sorted
                visited_states[s] = True
                nearest = nearest_visited_state(visited_list, s)
                q_update = q_table[nearest]
            else:
                q_update = q_table[s]

            if not visited_states[s_next]:
                bisect.insort(visited_list, s_next)  # insert while keeping sorted
                nearest = nearest_visited_state(visited_list, s_next)
                q_target = q_table[nearest]
            else:
                q_target = q_table[s_next]

            # if (task_name == 'medium' and state_to_discretized_vals(s + 1)[0] >= 457): # terminal state
            #     target = r
            # else: 
            target = r + gamma * np.max(q_target)

            q_table[s, a] = q_update[a] + alpha * (target - q_update[a])

    policy = np.zeros(n_states, dtype=int)

    # Derive policy using nearest visited state via binary search
    for i in range(n_states):
        if visited_states[i]:
            policy[i] = np.argmax(q_table[i])
        else:
            nearest = nearest_visited_state(visited_list, i)
            policy[i] = np.argmax(q_table[nearest])

    return policy + 1  # convert back to 1-indexed


def main():
    # get arguments from command line with defaults
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='small', help='Task name: small, medium, large')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--degree', type=int, default=None, help='Degree of polynomial features for medium task')
    args = parser.parse_args()

    if(args.task_name == 'small'):
        n_states = 100
        n_actions = 4
        input_data_path = './data/small.csv'
        output_path = 'small.policy'
        args.gamma = 0.95 if args.gamma is None else args.gamma
        args.num_epochs = 10 if args.num_epochs is None else args.num_epochs

    elif(args.task_name == 'medium'):
        n_states = 50000
        n_actions = 7
        input_data_path = './data/medium.csv'
        output_path = 'medium.policy'
        args.gamma = 1.0 if args.gamma is None else args.gamma
        args.num_epochs = 25 if args.num_epochs is None else args.num_epochs
        args.degree = 5 if args.degree is None else args.degree

    elif(args.task_name == 'large'):
        n_states = 302020
        n_actions = 9
        input_data_path = './data/large.csv'
        output_path = 'large.policy'
        args.gamma = 0.95 if args.gamma is None else args.gamma
        args.num_epochs = 50 if args.num_epochs is None else args.num_epochs

    # read data
    data = pd.read_csv(input_data_path)
    # convert all data to ints
    data = data.astype(int)
    if args.task_name == 'medium':
        print("Using specialized Q-learning for medium task")
        start_time = time()
        policy = fitted_q_iteration_poly(data, n_states, n_actions, gamma=args.gamma, n_iters=args.num_epochs, degree=args.degree)
        print(f"Medium task completed in {time() - start_time:.2f} seconds")
    else:
        print("Using general Q-learning")
        start_time = time()
        policy = q_learning(args.task_name, data, n_states, n_actions, alpha=args.alpha, gamma=args.gamma, epochs=args.num_epochs)
        print(f"General Q-learning on {args.task_name} completed in {time() - start_time:.2f} seconds")
    # output to a file 
    with open(output_path, 'w') as f:
        for _, action in enumerate(policy):
            f.write(f"{action}\n")

if __name__ == "__main__":
    main()
