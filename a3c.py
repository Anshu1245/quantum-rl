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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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


class CNNPolicyActorCritic(nn.Module):
    def __init__(self, num_qubits):
        super().__init__()
        self.num_qubits = num_qubits
        self.action_dim = 2 * num_qubits + 2 * (num_qubits - 1)

        self.shared_net = nn.Sequential(
            layer_init(nn.Conv2d(4, 128, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            layer_init(nn.Conv2d(128, 512, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU()
        )

        self.policy_head = layer_init(nn.Linear(128, self.action_dim), std=0.01)
        self.value_head = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, x):
        features = self.shared_net(x)
        return self.policy_head(features), self.value_head(features)

    def act(self, x, temperature=1.0):
        logits, value = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return action, value, logits


def worker(global_model, optimizer, worker_id, args, action_map):
    os.environ['OMP_NUM_THREADS'] = '1'  # Each worker single-threaded
    torch.manual_seed(worker_id)

    env_steps = args['STEPS_PER_EP']
    gamma = args['GAMMA']
    num_qubits = args['NUM_QUBITS']
    d = 1
    temp = 1.0
    eval_interval = 20         # Evaluate every 20 training episodes
    EVAL_EPS = 10              # Number of evaluation episodes
    EVAL_STEPS = 200           # Max steps per evaluation episode
    success_threshold = 0.8    # Success threshold to increase d


    device = torch.device("cpu")
    local_model = CNNPolicyActorCritic(num_qubits)
    local_model.load_state_dict(global_model.state_dict())

    for episode in range(args['NUM_EPISODES'] // args['NUM_WORKERS']):
        circuit = init_circuit(action_map, d, num_qubits)
        state = circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits).astype(np.float32)
        state = torch.Tensor(state).to(device)

        log_probs = []
        values = []
        rewards = []

        temp -= 1 / args['NUM_EPISODES']

        for step in range(env_steps):
            action, value, logits = local_model.act(state, temperature=temp)
            qc = quantum_circuit(num_qubits, action_map[action.item()][0], action_map[action.item()][1])
            c_ = Clifford(qc)
            next_circuit = circuit.compose(c_)
            next_state = next_circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits).astype(np.float32)

            reward = -1 if action.item() < 2*num_qubits else -10
            done = False
            operator = next_state
            if (operator.reshape((2*num_qubits, 2*num_qubits)) == np.identity(2*num_qubits)).all():
                done = True
                reward = 100

            next_state = torch.Tensor(next_state).to(device)

            probs = torch.softmax(logits, dim=-1)
            log_prob = torch.log(probs.gather(1, action))

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            circuit = next_circuit

            if done:
                break

        # Compute returns
        R = torch.zeros(1, 1).to(device)
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        # Loss
        actor_loss = -(log_probs.squeeze() * advantage.squeeze().detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()

        local_model.load_state_dict(global_model.state_dict())

        if worker_id == 0:
          print(f"[Worker {worker_id}] Episode {episode} | Loss: {loss.item():.4f}", flush=True)

    # --- EVALUATION PHASE ---
    if episode % eval_interval == 0 and episode > 0:
        success_rate = 0
        with torch.no_grad():
            for eval_ep in range(EVAL_EPS):
                circuit = init_circuit(action_map, d, num_qubits)
                state = circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits).astype(np.float32)
                state = torch.Tensor(state).to(device)

                for step in range(EVAL_STEPS):
                    logits, _ = local_model.forward(state)
                    action = torch.argmax(logits, dim=1, keepdim=True)

                    qc = quantum_circuit(num_qubits, action_map[action.item()][0], action_map[action.item()][1])
                    c_ = Clifford(qc)
                    next_circuit = circuit.compose(c_)
                    next_state = next_circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits).astype(np.float32)

                    operator = next_state
                    if (operator.reshape((2*num_qubits, 2*num_qubits)) == np.identity(2*num_qubits)).all():
                        success_rate += 1
                        break

                    next_state = torch.Tensor(next_state).to(device)
                    state = next_state
                    circuit = next_circuit

        success_rate /= EVAL_EPS
        print(f"[Worker {worker_id}] Evaluation success rate: {success_rate:.2f}", flush=True)

        if success_rate >= success_threshold:
            d += 1
            temp = 1.0
            print(f"[Worker {worker_id}] Updated d to {d}.........................", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args, unknown = parser.parse_known_args()  # <-- ignore unknown Jupyter flags

    NUM_WORKERS = args.num_workers


    # Hyperparameters
    NUM_EPISODES = 10000
    STEPS_PER_EP = 200
    GAMMA = 0.99
    LR = 1e-4
    NUM_QUBITS = 7
    DEVICE = torch.device("cpu")

    print(f"Running A3C with {NUM_WORKERS} workers...")

    action_map = create_action_map(NUM_QUBITS)
    global_model = CNNPolicyActorCritic(NUM_QUBITS).to(DEVICE)
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=LR)

    args = {
        'NUM_EPISODES': NUM_EPISODES,
        'STEPS_PER_EP': STEPS_PER_EP,
        'GAMMA': GAMMA,
        'NUM_WORKERS': NUM_WORKERS,
        'NUM_QUBITS': NUM_QUBITS,
    }

    processes = []
    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(global_model, optimizer, worker_id, args, action_map))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), "a3c_model.pth")
    print("Training Complete!")

