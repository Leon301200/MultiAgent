import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt


class GridEnvironment:
    def __init__(self, size=10):
        self.size = size  # Taille de la grille
        self.state_size = 4  # Taille de l'état (position du policier et du voleur)
        self.reset()  # Initialisation de l'environnement

    def reset(self):
        # Position initiale aléatoire du policier et du voleur
        self.police_position = np.random.randint(0, self.size, size=2)
        self.thief_position = np.random.randint(0, self.size, size=2)

        self.step_counter = 0  # Compteur de pas réinitialisé

        # S'assurer que le policier et le voleur ne commencent pas au même endroit
        while np.array_equal(self.police_position, self.thief_position):
            self.thief_position = np.random.randint(0, self.size, size=2)

        return self.get_state()

    def step(self, police_action, thief_action):
        # Increment step counter
        self.step_counter += 1

        # Positions avant de faire le mouvement
        self.previous_police_position = np.copy(self.police_position)
        self.previous_thief_position = np.copy(self.thief_position)

        # Mettre à jour les position en fonction de l'action choisi
        self.police_position = self.update_position(self.police_position, police_action)
        self.thief_position = self.update_position(self.thief_position, thief_action)

        police_reward, thief_reward, done = self.get_reward()

        return self.get_state(), (police_reward, thief_reward), done

    def get_state(self):
        # Flatten the positions into a single state vector
        return np.concatenate((self.police_position, self.thief_position))

    def update_position(self, position, action):
        # Defini les actions avec des vecteurs
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]

        # Verifie si la position est bien dans la grille
        new_position = position + actions[action]
        new_position = np.clip(new_position, 0, self.size-1)
        return new_position

    def get_reward(self):
        distance_before = np.linalg.norm(self.previous_police_position - self.previous_thief_position)
        distance_after = np.linalg.norm(self.police_position - self.thief_position)

        # Step penalty
        step_penalty = -0.05

        # Police and thief rewards
        police_reward = step_penalty
        thief_reward = step_penalty

        # Police gets closer or thief gets further
        if distance_after < distance_before:
            police_reward += 0.1
        if distance_after > distance_before:
            thief_reward += 0.1

        # Catching or escaping
        if np.array_equal(self.police_position, self.thief_position):
            return 5 + police_reward, -5 + thief_reward, True
        if self.step_counter >= 50:
            return -5 + police_reward, 5 + thief_reward, True

        return police_reward, thief_reward, False

    def render(self, screen, cell_size):
        # Define colors
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        BLACK = (0, 0, 0)

        # Clear the screen with white background
        screen.fill(WHITE)

        # Draw the grid lines
        for x in range(0, self.size * cell_size, cell_size):
            for y in range(0, self.size * cell_size, cell_size):
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(screen, BLACK, rect, 1)

        # Draw the police as a blue circle
        police_pos = (self.police_position[1] * cell_size + cell_size // 2,
                      self.police_position[0] * cell_size + cell_size // 2)
        pygame.draw.circle(screen, BLUE, police_pos, cell_size // 4)

        # Draw the thief as a red circle
        thief_pos = (self.thief_position[1] * cell_size + cell_size // 2,
                     self.thief_position[0] * cell_size + cell_size // 2)
        pygame.draw.circle(screen, RED, thief_pos, cell_size // 4)

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, model_base_filename=None, starting_episode=0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model_base_filename = model_base_filename
        self.current_episode = starting_episode
        if self.model_base_filename and starting_episode != 0:
            self.load_model(self.model_base_filename + f"_{self.current_episode}.pth")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target = reward if done else (reward + self.gamma * np.amax(self.target_model(next_state).detach().numpy()))
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.SmoothL1Loss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        if self.model_base_filename:
            filename = self.model_base_filename + f"_{self.current_episode}.pth"
            torch.save(self.model.state_dict(), filename)
            print(f"Model saved to {filename}")
        else:
            print("Model base filename not provided. Model not saved.")

    def load_model(self, filename):
        try:
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()  # Set the model to evaluation mode
            print(f"Loaded model weights from {filename}")
        except FileNotFoundError:
            print(f"No saved model found for {filename}. Starting from scratch.")


def train_step(env, police_agent, thief_agent, batch_size):
    # Une étape d'entraînement pour les agents
    if env.done:  # Verifie si l'episode est terminer
        state = env.reset()  # Reset l'envirronement
        # print(f"Starting a new episode")
    else:
        state = env.get_state()  # Continue depuis le dernier etat

    state = np.reshape(state, [1, env.state_size])

    # Agents choisi une action
    police_action = police_agent.act(state)
    thief_action = thief_agent.act(state)

    # Environment takes a step based on the actions
    next_state, (police_reward, thief_reward), done = env.step(police_action, thief_action)
    next_state = np.reshape(next_state, [1, env.state_size])

    # Agents remember the experience
    police_agent.remember(state, police_action, police_reward, next_state, done)
    thief_agent.remember(state, thief_action, thief_reward, next_state, done)

    # Replay si la memoire est suffisante
    if len(police_agent.memory) > batch_size:
        police_agent.replay(batch_size)
    if len(thief_agent.memory) > batch_size:
        thief_agent.replay(batch_size)

    env.done = done  # Update the done status of the environment

    # print(f"Episode ended: Police Reward: {police_reward}, Thief Reward: {thief_reward}, Step : {env.step_counter}")

    # if done:
    #     print(f"Episode ended: Police Reward: {police_reward}, Thief Reward: {thief_reward}, Step : {env.step_counter}")

    return done, police_reward, thief_reward, env.step_counter

def run_episode(env, police_agent, thief_agent, verbose=False):
    """ Run a single episode and return the outcome and steps taken. """
    state = env.reset()
    done = False
    steps = 0

    while not done:
        steps += 1
        police_action = police_agent.act(state)
        thief_action = thief_agent.act(state)
        next_state, (police_reward, thief_reward), done = env.step(police_action, thief_action)
        state = next_state

        if done and verbose:
            print(f"Episode ended: Police Reward: {police_reward}, Thief Reward: {thief_reward}, Steps: {steps}")

    return police_reward, thief_reward, steps

def test_model(env, police_agent, thief_agent, episodes=100, test_type='random'):
    """ Test the model over a specified number of episodes. """
    win_loss = {'police': 0, 'thief': 0}
    total_steps = 0

    for _ in range(episodes):
        if test_type == 'fixed':
            # Set fixed initial conditions
            env.police_position = np.array([0, 0])
            env.thief_position = np.array([env.size - 1, env.size - 1])

        police_reward, thief_reward, steps = run_episode(env, police_agent, thief_agent)
        total_steps += steps

        if police_reward > 0:
            win_loss['police'] += 1
        else:
            win_loss['thief'] += 1

    avg_steps = total_steps / episodes
    return win_loss, avg_steps

# Initialize Pygame, environment, and agents
pygame.init()
size = 10
cell_size = 60
screen_size = size * cell_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Police and Thief Game")

max_episodes = 10000  # Set the maximum number of episodes

env = GridEnvironment(size=size)
state_size = 4  # Update based on your state representation
action_size = 9  # Assuming 8 possible directions + stay

police_agent = DQNAgent(state_size, action_size, 'model/police_agent_model', starting_episode=10000)
thief_agent = DQNAgent(state_size, action_size, 'model/thief_agent_model', starting_episode=10000)

columns = ['Episode', 'Police_Reward', 'Thief_Reward', 'Steps', 'Epsilon', 'Winner']
episode_data = pd.DataFrame(columns=columns)
total_police_reward = 0
total_thief_reward = 0

batch_size = 32
env.done = True

episode_count = 0
save_interval = 100

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    done, police_reward, thief_reward, steps = train_step(env, police_agent, thief_agent, batch_size)

    # Render the current state
    env.render(screen, cell_size)
    pygame.display.flip()
    pygame.time.wait(100)  # Control the speed of the loop

    if done:
        episode_count += 1
        police_agent.current_episode += 1
        thief_agent.current_episode += 1
        print(f"Episode: {episode_count} / {max_episodes}")

        # Determine the winner
        winner = 'Thief' if steps >= 50 else 'Police'

        # Log data
        new_row = pd.DataFrame([{
            'Episode': episode_count,
            'Police_Reward': round(total_police_reward, 2),
            'Thief_Reward': round(total_thief_reward, 2),
            'Steps': steps,
            'Epsilon': police_agent.epsilon,
            'Winner' : winner
        }])

        episode_data = pd.concat([episode_data, new_row], ignore_index=True)

        total_police_reward = 0
        total_thief_reward = 0

        if episode_count % save_interval == 0:  # Test every 10 episodes
            win_loss, avg_steps = test_model(env, police_agent, thief_agent, episodes=10)
            print(f"Test Results - Police Wins: {win_loss['police']}, Thief Wins: {win_loss['thief']}, Average Steps: {avg_steps:.2f}")

            # Save episode data to CSV
            episode_data.to_csv('episode_performance.csv', index=False)

            # Save model
            police_agent.save_model()
            thief_agent.save_model()
    else:
        total_police_reward += police_reward
        total_thief_reward += thief_reward

    if episode_count >= max_episodes:
        print(f"Reached the maximum of {max_episodes} episodes. Stopping training.")
        running = False  # Exit the loop
pygame.quit()
