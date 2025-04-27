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

# --- Quantum Environment Utilities ---

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

# --- Neural Networks ---

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNActor(nn.Module):
    def __init__(self, num_qubits, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Conv2d(4, 128, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            layer_init(nn.Conv2d(128, 512, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim))
        )
    
    def forward(self, x):
        logits = self.shared(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

class CNNCritic(nn.Module):
    def __init__(self, num_qubits, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Conv2d(4, 128, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            layer_init(nn.Conv2d(128, 512, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )
        self.q_head = layer_init(nn.Linear(256, action_dim))

    def forward(self, x):
        features = self.shared(x)
        return self.q_head(features)

# --- Worker Process ---

def worker(actor_model, queue, worker_id, args, action_map):
    print(f"Worker {worker_id} started", flush=True)
    device = torch.device("cpu")
    num_qubits = args['NUM_QUBITS']
    d = 1

    for episode in range(args['MAX_EPISODES'] // args['NUM_WORKERS']):
        circuit = init_circuit(action_map, d, num_qubits)
        state = torch.Tensor(circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

        for step in range(args['MAX_STEPS']):
            with torch.no_grad():
                probs = actor_model(state)
                action = torch.multinomial(probs, num_samples=1)

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

# --- Evaluation Function ---

def evaluate(actor, action_map, d, num_qubits, eval_episodes=10, eval_steps=200):
    device = torch.device("cpu")
    success_count = 0

    actor.eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            circuit = init_circuit(action_map, d, num_qubits)
            state = torch.Tensor(circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

            for _ in range(eval_steps):
                probs = actor(state)
                action = torch.argmax(probs, dim=1, keepdim=True)

                qc = quantum_circuit(num_qubits, action_map[action.item()][0], action_map[action.item()][1])
                c_ = Clifford(qc)
                next_circuit = circuit.compose(c_)
                next_state = torch.Tensor(next_circuit.symplectic_matrix.reshape(1, 4, num_qubits, num_qubits)).float().to(device)

                if (next_state.cpu().numpy().reshape(2*num_qubits, 2*num_qubits) == np.identity(2*num_qubits)).all():
                    success_count += 1
                    break

                state = next_state
                circuit = next_circuit

    actor.train()
    return success_count / eval_episodes

# --- Learner Process ---

def learner(actor, critic1, critic2, target_critic1, target_critic2, optimizer_actor, optimizer_critic, queue, args, action_map):
    print("Learner started", flush=True)
    device = torch.device("cpu")
    gamma = args['GAMMA']
    tau = args['TAU']
    buffer = deque(maxlen=100_000)
    d = 1
    eval_interval = 1
    success_threshold = 0.8

    for step in range(args['MAX_GRAD_STEPS']):
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

        # Critic update
        with torch.no_grad():
            next_probs = actor(next_states)
            next_actions = torch.multinomial(next_probs, num_samples=1)
            next_q1 = target_critic1(next_states).gather(1, next_actions)
            next_q2 = target_critic2(next_states).gather(1, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + gamma * (1 - dones) * next_q

        current_q1 = critic1(states).gather(1, actions)
        current_q2 = critic2(states).gather(1, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # Actor update
        probs = actor(states)
        sampled_actions = torch.multinomial(probs, num_samples=1)
        q1 = critic1(states).gather(1, sampled_actions)
        q2 = critic2(states).gather(1, sampled_actions)
        q = torch.min(q1, q2)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
        actor_loss = (-q - args['ALPHA'] * entropy).mean()

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if step % 100 == 0:
            print(f"Learner Step {step}: Actor Loss {actor_loss.item():.4f}, Critic Loss {critic_loss.item():.4f}", flush=True)

        if step % eval_interval == 0 and step > 0:
            success_rate = evaluate(actor, action_map, d, args['NUM_QUBITS'])
            print(f"Eval at step {step}: Success Rate {success_rate:.2f}", flush=True)
            if success_rate >= success_threshold:
                d += 1
                print(f"Difficulty increased to {d} at step {step}!", flush=True)

# --- Main Function ---

if __name__ == "__main__":
    import argparse
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args, unknown = parser.parse_known_args()

    NUM_WORKERS = args.num_workers
    NUM_QUBITS = 4
    ACTION_DIM = 2 * NUM_QUBITS + 2 * (NUM_QUBITS - 1)

    actor = CNNActor(NUM_QUBITS, ACTION_DIM)
    critic1 = CNNCritic(NUM_QUBITS, ACTION_DIM)
    critic2 = CNNCritic(NUM_QUBITS, ACTION_DIM)
    target_critic1 = CNNCritic(NUM_QUBITS, ACTION_DIM)
    target_critic2 = CNNCritic(NUM_QUBITS, ACTION_DIM)

    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor.share_memory()
    critic1.share_memory()
    critic2.share_memory()

    optimizer_actor = optim.Adam(actor.parameters(), lr=3e-4)
    optimizer_critic = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4)

    queue = mp.Queue()
    action_map = create_action_map(NUM_QUBITS)

    args = {
        'NUM_QUBITS': NUM_QUBITS,
        'MAX_EPISODES': 10000,
        'MAX_STEPS': 200,
        'MAX_GRAD_STEPS': 50000,
        'BATCH_SIZE': 256,
        'GAMMA': 0.99,
        'TAU': 0.005,
        'ALPHA': 0.2,
        'NUM_WORKERS': NUM_WORKERS
    }

    processes = []

    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(actor, queue, worker_id, args, action_map))
        p.start()
        processes.append(p)

    p = mp.Process(target=learner, args=(actor, critic1, critic2, target_critic1, target_critic2, optimizer_actor, optimizer_critic, queue, args, action_map))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    torch.save(actor.state_dict(), "apex_sac_actor.pth")
    print("Training Complete!")
