import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

# --- Utilities for Environment ---

def create_action_map(num_qubits):
    action_map = {}
    for h in range(num_qubits):
        action_map[h] = ("H", h)
    for s in range(num_qubits, 2 * num_qubits):
        action_map[s] = ("S", s - num_qubits)
    cx_index = 2 * num_qubits
    for control in range(num_qubits-1):
        action_map[cx_index] = ("CX", [control, control+1])
        action_map[cx_index+num_qubits-1] = ("CX", [control+1, control])
        cx_index += 1
    return action_map

def quantum_circuit(num_qubits, gate_type, gate_qubits):
    qc = QuantumCircuit(num_qubits)
    if gate_type == "H":
        qc.h(gate_qubits)
    elif gate_type == "S":
        qc.s(gate_qubits)
    elif len(gate_qubits) == 2:
        qc.cx(gate_qubits[0], gate_qubits[1])
    return qc

def init_circuit(action_map, d, num_qubits):
    qc = QuantumCircuit(num_qubits)
    target_sequence = random.choices(list(action_map.values()), k=d)
    for gate in target_sequence:
        qc.append(quantum_circuit(num_qubits, gate[0], gate[1]), range(num_qubits))
    return Clifford(qc)

# --- DQN Model ---

def layer_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0)
    return layer

class DQN(nn.Module):
    def __init__(self, num_qubits, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(4, 128, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            layer_init(nn.Conv2d(128, 512, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            layer_init(nn.Linear(512 * (num_qubits//4) * (num_qubits//4), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, action_dim))
        )
        
    def forward(self, x):
        return self.net(x)

# --- Worker Process ---

def worker(policy_net, queue, worker_id, args, action_map):
    print(f"Worker {worker_id} started", flush=True)
    device = torch.device("cpu")
    num_qubits = args['NUM_QUBITS']
    epsilon = args['EPSILON_START']
    d = 1

    for episode in range(args['MAX_EPISODES'] // args['NUM_WORKERS']):
        circuit = init_circuit(action_map, d, num_qubits)
        state = torch.Tensor(circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

        for step in range(args['MAX_STEPS']):
            with torch.no_grad():
                if random.random() < epsilon:
                    action = torch.randint(0, args['ACTION_DIM'], (1,1))
                else:
                    q_values = policy_net(state)
                    action = q_values.argmax(dim=1, keepdim=True)

            qc = quantum_circuit(num_qubits, action_map[action.item()][0], action_map[action.item()][1])
            c_ = Clifford(qc)
            next_circuit = circuit.compose(c_)
            next_state = torch.Tensor(next_circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

            reward = -1 if action.item() < 2*num_qubits else -10
            done = False
            operator = next_state
            if (operator.cpu().numpy().reshape(2*num_qubits, 2*num_qubits) == np.identity(2*num_qubits)).all():
                done = True
                reward = 100

            queue.put((state.cpu().numpy(), action.item(), reward, next_state.cpu().numpy(), done))

            if done:
                break

            state = next_state
            circuit = next_circuit

        epsilon = max(args['EPSILON_END'], epsilon * args['EPSILON_DECAY'])

# --- Evaluation ---

def evaluate(policy_net, action_map, d, num_qubits, eval_episodes=10, eval_steps=200):
    device = torch.device("cpu")
    success_count = 0
    policy_net.eval()

    with torch.no_grad():
        for _ in range(eval_episodes):
            circuit = init_circuit(action_map, d, num_qubits)
            state = torch.Tensor(circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

            for _ in range(eval_steps):
                q_values = policy_net(state)
                action = q_values.argmax(dim=1, keepdim=True)

                qc = quantum_circuit(num_qubits, action_map[action.item()][0], action_map[action.item()][1])
                c_ = Clifford(qc)
                next_circuit = circuit.compose(c_)
                next_state = torch.Tensor(next_circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

                if (next_state.cpu().numpy().reshape(2*num_qubits, 2*num_qubits) == np.identity(2*num_qubits)).all():
                    success_count += 1
                    break

                state = next_state
                circuit = next_circuit

    policy_net.train()
    return success_count / eval_episodes

# --- Learner Process ---

def learner(policy_net, target_net, optimizer, queue, args, action_map):
    print("Learner started", flush=True)
    device = torch.device("cpu")
    buffer = deque(maxlen=args['REPLAY_SIZE'])
    d = 1
    success_threshold = 0.8
    eval_interval = 1
    steps = 0

    while steps < args['MAX_GRAD_STEPS']:
        while not queue.empty():
            buffer.append(queue.get())

        if len(buffer) < args['BATCH_SIZE']:
            continue

        batch = random.sample(buffer, args['BATCH_SIZE'])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.Tensor(np.concatenate(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.Tensor(np.concatenate(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_q = target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + args['GAMMA'] * (1 - dones) * next_q

        current_q = policy_net(states).gather(1, actions)

        loss = nn.MSELoss()(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % args['TARGET_UPDATE'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if steps % 100 == 0:
            print(f"Learner Step {steps}: Loss {loss.item():.4f}", flush=True)

        if steps % eval_interval == 0 and steps > 0:
            success_rate = evaluate(policy_net, action_map, d, args['NUM_QUBITS'])
            print(f"Eval at step {steps}: Success Rate {success_rate:.2f}", flush=True)
            if success_rate >= success_threshold:
                d += 1
                print(f"Difficulty increased to {d} at step {steps}!", flush=True)

        steps += 1

# --- Main ---

if __name__ == "__main__":
    import argparse
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args, unknown = parser.parse_known_args()

    NUM_WORKERS = args.num_workers
    NUM_QUBITS = 4
    ACTION_DIM = 2 * NUM_QUBITS + 2 * (NUM_QUBITS - 1)

    policy_net = DQN(NUM_QUBITS, ACTION_DIM)
    target_net = DQN(NUM_QUBITS, ACTION_DIM)
    target_net.load_state_dict(policy_net.state_dict())

    policy_net.share_memory()
    target_net.share_memory()

    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)

    queue = mp.Queue()
    action_map = create_action_map(NUM_QUBITS)

    args = {
        'NUM_QUBITS': NUM_QUBITS,
        'ACTION_DIM': ACTION_DIM,
        'MAX_EPISODES': 10000,
        'MAX_STEPS': 200,
        'MAX_GRAD_STEPS': 50000,
        'BATCH_SIZE': 256,
        'GAMMA': 0.99,
        'REPLAY_SIZE': 100_000,
        'TARGET_UPDATE': 100,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 0.995,
        'NUM_WORKERS': NUM_WORKERS
    }

    processes = []

    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(policy_net, queue, worker_id, args, action_map))
        p.start()
        processes.append(p)

    p = mp.Process(target=learner, args=(policy_net, target_net, optimizer, queue, args, action_map))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    torch.save(policy_net.state_dict(), "parallel_dqn_policy.pth")
    print("Training Complete!")
