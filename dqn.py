import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNPolicyDQN(nn.Module):
    def __init__(self, num_qubits):
        super().__init__()
        self.num_qubits = num_qubits
        self.action_dim = 2 * num_qubits + 2 * (num_qubits - 1)  # Total actions in DQN

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            layer_init(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.action_dim)),  # Output Q-values for all actions
        )


    def forward(self, x):
        return self.network(x)  # Returns Q-values for all actions

    def get_q_values(self, x):
        return self.forward(x)  # Same as forward pass

    def select_action_eps_greedy(self, state, epsilon):
        """Select action using an epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return torch.tensor([[np.random.randint(0, self.action_dim)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.forward(state).argmax(dim=1, keepdim=True)  # Greedy action selection

    def select_action_softmax(self, state, temperature=1.0):
      """Select action using Softmax (Boltzmann) policy."""
      with torch.no_grad():
          q_values = self.forward(state)  # Shape: (batch_size, action_dim)
          # Scale Q-values by temperature
          probs = torch.softmax(q_values / temperature, dim=1)  # Softmax over actions
          # Sample an action based on the probabilities
          action = torch.multinomial(probs, num_samples=1)
          return action  # Shape: (batch_size, 1)

    def select_action_greedy(self, state):
      """Select action using a purely greedy policy (no exploration)."""
      with torch.no_grad():
          return self.forward(state).argmax(dim=1, keepdim=True)  # Greedy action selection


# Function to compute the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = CNNPolicyDQN(num_qubits=7)
# print(count_parameters(model))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


def quantum_circuit(num_qubits, gate_type, gate_qubits):
  qc = QuantumCircuit(num_qubits)
  if gate_type=="H":
    assert type(gate_qubits)==int
    qc.h(gate_qubits)
  elif gate_type=="S":
    assert type(gate_qubits)==int
    qc.s(gate_qubits)
  elif len(gate_qubits)==2:
    qc.cx(gate_qubits[0], gate_qubits[1])
  return qc



def create_action_map(num_qubits):
  action_map = {}

  # H gates: first num_qubits keys
  for h in range(num_qubits):
      action_map[h] = ("H", h)  # H gate acts on a single qubit

  # S gates: next num_qubits keys
  for s in range(num_qubits, 2 * num_qubits):
      action_map[s] = ("S", s - num_qubits)  # S gate acts on a single qubit

  # CX gates: last 2 * (num_qubits - 1) keys
  cx_index = 2 * num_qubits  # Start index for CX gates
  for control in range(num_qubits-1):
      action_map[cx_index] = ("CX", [control, control+1])  # CX gate acts on two qubits
      action_map[cx_index+num_qubits-1] = ("CX", [control+1, control])  # CX gate acts on two qubits
      cx_index += 1

  return action_map


def init_circuit(action_map, d, num_qubits):
  qc = QuantumCircuit(num_qubits)
  target_sequence = random.choices(list(action_map.values()), k=d)
  for gate in target_sequence:
    gate_type = gate[0]
    gate_qubits = gate[1]
    if gate_type=="H":
      assert type(gate_qubits)==int
      qc.h(gate_qubits)
    elif gate_type=="S":
      assert type(gate_qubits)==int
      qc.s(gate_qubits)
    elif len(gate_qubits)==2:
      qc.cx(gate_qubits[0], gate_qubits[1])
  return Clifford(qc)


# Hyperparameters
NUM_EPISODES = 10000
STEPS_PER_EP = 200
EVAL_EPS = 50
EVAL_STEPS = 200
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
LR = 1e-4  # Learning rate
EPSILON_START = 1.0  # Initial epsilon for exploration
EPSILON_END = 0.05  # Final epsilon
EPSILON_DECAY = 500  # Decay rate
TARGET_UPDATE_FREQ = 50  # How often to update target network
MEMORY_SIZE = 500000  # Replay buffer size
NUM_QUBITS = 7  # Number of qubits
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


action_map = create_action_map(NUM_QUBITS)
print(action_map)

# Initialize policy and target networks
policy_net = CNNPolicyDQN(NUM_QUBITS).to(DEVICE)
target_net = CNNPolicyDQN(NUM_QUBITS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())  # Copy initial weights
target_net.eval()  # Target net in evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=MEMORY_SIZE)

# Epsilon decay schedule
def get_epsilon(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY)

# Function to train DQN using replay buffer
def train_dqn():
    if len(replay_buffer) < BATCH_SIZE:
        return  # Don't train until enough samples

    batch = random.sample(replay_buffer, BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.cat(state_batch).to(DEVICE)
    action_batch = torch.cat(action_batch).to(DEVICE)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_state_batch = torch.cat(next_state_batch).to(DEVICE)
    done_batch = torch.tensor(done_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Compute Q-values for selected actions
    q_values = policy_net.get_q_values(state_batch).gather(1, action_batch)

    # Compute target Q-values using target network
    with torch.no_grad():
        next_q_values = target_net.get_q_values(next_state_batch).max(1, keepdim=True)[0]
        target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # Gradient clipping
    optimizer.step()

# Training Loop
d = 1
temp = 1.0
for episode in range(NUM_EPISODES):
    circuit = init_circuit(action_map, d, NUM_QUBITS)
    state = circuit.symplectic_matrix.reshape(1, 4, NUM_QUBITS, NUM_QUBITS).astype(np.float32)
    state = torch.Tensor(state).to(DEVICE)
    total_reward = 0

    temp -= 1 / NUM_EPISODES
    for step in range(STEPS_PER_EP):  # Max steps per episode
        epsilon = get_epsilon(episode)
        action = policy_net.select_action_softmax(state, temperature=temp).to(DEVICE)

        # print(action_map[action.item()][0], action_map[action.item()][1])
        qc = quantum_circuit(NUM_QUBITS, action_map[action.item()][0], action_map[action.item()][1])
        c_ = Clifford(qc)
        next_circuit = circuit.compose(c_)
        next_state = next_circuit.symplectic_matrix.reshape(1, 4, NUM_QUBITS, NUM_QUBITS).astype(np.float32)

        if action.item() < 2*NUM_QUBITS:
            reward = -1
        else:
            reward = -10

        done = False
        # print(next_state)
        operator = next_state
        # if episode % 2 == 0 and step == 100:
        #     print(operator.reshape((2*NUM_QUBITS, 2*NUM_QUBITS)), "\n", np.identity(2*NUM_QUBITS))
        if (operator.reshape((2*NUM_QUBITS, 2*NUM_QUBITS)) == np.identity(2*NUM_QUBITS)).all():
            print(f"Epsiode {episode}")
            print("yaaaaaaaaay!")
            done = True
            reward = 100

        next_state = torch.Tensor(next_state).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float32, device=DEVICE)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        circuit = next_circuit
        total_reward += reward.item()

        train_dqn()  # Train at each step

        if done:
            break

    # eval
    success_rate = 0
    for eval_ep in range(EVAL_EPS):
        with torch.no_grad():
            circuit = init_circuit(action_map, d, NUM_QUBITS)
            state = circuit.symplectic_matrix.reshape(1, 4, NUM_QUBITS, NUM_QUBITS).astype(np.float32)
            state = torch.Tensor(state).to(DEVICE)

            for step in range(EVAL_STEPS):
                action = policy_net.select_action_greedy(state).to(DEVICE)
                qc = quantum_circuit(NUM_QUBITS, action_map[action.item()][0], action_map[action.item()][1])
                c_ = Clifford(qc)
                next_circuit = circuit.compose(c_)
                next_state = next_circuit.symplectic_matrix.reshape(1, 4, NUM_QUBITS, NUM_QUBITS).astype(np.float32)

                done = False
                operator = next_state
                if (operator.reshape((2*NUM_QUBITS, 2*NUM_QUBITS)) == np.identity(2*NUM_QUBITS)).all():
                    done = True
                    success_rate += 1
                    break

                next_state = torch.Tensor(next_state).to(DEVICE)
                state = next_state
                circuit = next_circuit

    print(success_rate / EVAL_EPS)
    if success_rate / EVAL_EPS >= 0.8:
      d += 1
      print(f"Updated d to {d}.........................")
      # print(f"temperature was {temp}")
      # temp = 1.0
      # print(f"temperature is now {temp}")

    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())


    # print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {get_epsilon(episode):.4f}")

torch.save(policy_net.state_dict(), "dqn_model.pth")
print("Training Complete!")

