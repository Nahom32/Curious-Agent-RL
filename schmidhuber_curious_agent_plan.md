# Schmidhuber's 1991 Curious Model-Building Control System
## Implementation Plan for Q-Learning / Deep Q-Learning Experiment

> Based on: Jürgen Schmidhuber, *"Curious Model-Building Control Systems"*,
> Proc. International Joint Conference on Neural Networks, Singapore, 1991.
> [HTML version](https://people.idsia.ch/~juergen/curioussingapore/curioussingapore.html)

---

## 1. Architecture Overview

The system consists of **4 interacting modules**:

| Module | Role | What It Learns |
|--------|------|----------------|
| **M** — World Model | Predicts `x(t+1)` from `(x(t), a(t))` | Environmental dynamics |
| **C** — Confidence Network | Predicts M's expected error | "How wrong will M be?" |
| **Q** — Q-Function | Evaluates state-action pairs | Expected cumulative reward |
| **A** — Action Policy | Chooses actions | Greedy over Q (with ε-exploration) |

---

## 2. The Core Curiosity Mechanism

This is the key insight from the 1991 paper. Unlike naive curiosity (rewarding high prediction error), Schmidhuber's agent rewards **improvement in prediction ability**:

```
r_curiosity(t) = o_C_before(t) − o_C_after(t)
```

- **Before** taking action, C predicts how much error M will make
- M sees the actual transition and updates
- C is then updated on the actual error
- **Curiosity reward** = how much C's error prediction *dropped* for that same input

This means:
- ✅ **Deterministic, learnable zones** → M improves → C's error prediction drops → **positive reward**
- ❌ **Noisy zones** → M can't improve → C stays high → **no sustained reward**
- 😴 **Already-known zones** → M already perfect → C near zero → **no reward** (boredom)

The agent is attracted to the **"zone of proximal learning"** — not too easy, not too hard.

---

## 3. Proposed Pygame Environment: "Curiosity Grid World"

### 3.1 Environment Design

A 10×10 grid (100 discrete states) with 4 action directions (↑↓←→). The agent sees its `(row, col)` as a 2D vector.

**Five zone types** (visually distinct colors):

| Zone | Behavior | Why It Matters |
|------|----------|----------------|
| **Static** | `next_state = apply(action)` with no zone modifier | Boring, predictable baseline |
| **Deterministic-A** | `next_state = shift_A(apply(action))` | Primary learning target |
| **Deterministic-B** | `next_state = shift_B(apply(action))` | Secondary target — harder or different |
| **Noisy** | `next_state = perturb(apply(action))` | Partly unlearnable — agent should avoid |
| **Dynamic** | Modifier switches between A and B every N steps | Tests re-curiosity |

**External reward** `r_ext`: Place a small goal in one corner. The agent gets +1 for reaching it. This tests whether curiosity *helps* or *hurts* task completion.

### 3.2 State Representation

- **Tabular Q**: `(row, col)` as discrete state index
- **Deep Q**: One-hot or coordinate encoding → small MLP

---

## 4. Phase 1: Tabular Q-Learning (Minimal Viable Experiment)

**Goal**: Validate the curiosity mechanism with lookup tables.

```
M: Lookup table  M[state][action] → predicted_next_state
C: Lookup table  C[state][action] → predicted_error
Q: Lookup table  Q[state][action] → value
A: Derived from Q (ε-greedy)
```

### 4.1 Algorithm per Timestep

```python
# 1. Observe current state
state = env.get_state()  # (row, col)
state_idx = state_to_idx(state)

# 2. Get C's confidence BEFORE action
c_input = encode(state_idx, action)  # or just (state_idx, action) pair
o_c_before = C.predict(c_input)      # expected error for this transition

# 3. Choose action (ε-greedy)
if random() < epsilon:
    action = random_action()
else:
    action = argmax_a Q[state_idx][a]

# 4. Execute action, observe outcome
next_state, r_ext, done = env.step(action)
next_state_idx = state_to_idx(next_state)

# 5. Compute actual model error
predicted_next = M.predict(state_idx, action)
actual_error = mse(predicted_next, next_state)

# 6. Update Model M
M.update(state_idx, action, next_state, lr=alpha_m)

# 7. Update Confidence C (now it knows the actual error)
C.update(c_input, target=actual_error, lr=alpha_c)
o_c_after = C.predict(c_input)   # C's new prediction for SAME input

# 8. Compute curiosity reward
r_curiosity = o_c_before - o_c_after   # KEY: improvement = reward
r_total = r_ext + beta * r_curiosity   # beta scales curiosity vs external goal

# 9. Update Q (Watkins' Q-learning)
best_next_q = max(Q[next_state_idx])
td_target = r_total + gamma * best_next_q
td_error = td_target - Q[state_idx][action]
Q[state_idx][action] += alpha_q * td_error

# 10. Decay epsilon
epsilon *= epsilon_decay
```

### 4.2 Hyperparameters

| Parameter | Description | Suggested Range |
|-----------|-------------|-----------------|
| `alpha_m` | Model learning rate | 0.1 – 0.3 |
| `alpha_c` | Confidence learning rate | 0.05 – 0.1 (slower than M!) |
| `alpha_q` | Q-learning rate | 0.1 – 0.2 |
| `beta` | Curiosity weight | 0.5 – 2.0 |
| `gamma` | Discount factor | 0.9 – 0.99 |
| `epsilon` | Exploration rate | 1.0 → 0.01 |

> **Why C learns slower than M**: C needs to track M's *expected* error, not react to every single sample. A slower learning rate makes C a stable meta-predictor.

---

## 5. Phase 2: Deep Q-Learning (DQN) Version

Replace lookup tables with small neural networks.

```
M:  MLP(input=state+action, hidden=[64,64], output=next_state)
C:  MLP(input=state+action, hidden=[32,32], output=scalar_error)
Q:  MLP(input=state+action, hidden=[128,128], output=Q_values_per_action)
    OR: Dueling DQN architecture
A:  argmax over Q-network outputs
```

### 5.1 Key Differences from Standard DQN

1. **M and C are trained online** (not from replay buffer) — they need to track the current model of the environment
2. **Curiosity reward is computed per-step** using C's before/after predictions
3. **Q-network trains on `(r_ext + r_curiosity)`** instead of just `r_ext`

### 5.2 Training Loop (Simplified)

```python
for episode in range(num_episodes):
    state = env.reset()

    for t in range(max_steps):
        # Curiosity: C predicts error BEFORE action
        state_tensor = torch.FloatTensor(state)
        q_values = Q_net(state_tensor)

        # ε-greedy
        if random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = q_values.argmax().item()

        # C's prediction before seeing outcome
        sa = torch.cat([state_tensor, one_hot(action)])
        o_c_before = C_net(sa).item()

        # Execute
        next_state, r_ext, done = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state)

        # Update M (world model)
        predicted_next = M_net(sa)
        m_loss = mse(predicted_next, next_state_tensor)
        m_optimizer.zero_grad()
        m_loss.backward()
        m_optimizer.step()

        # Update C (confidence)
        actual_error = m_loss.item()  # or |predicted - actual|
        c_loss = mse(C_net(sa), torch.tensor([actual_error]))
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        # C's prediction AFTER update (same input!)
        o_c_after = C_net(sa).item()

        # Curiosity reward
        r_curiosity = o_c_before - o_c_after
        r_total = r_ext + beta * r_curiosity

        # Store in replay buffer for Q
        replay_buffer.push(state, action, r_total, next_state, done)

        # Train Q (standard DQN mini-batch)
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            # ... standard DQN loss ...
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

        state = next_state
```

---

## 6. Phase 3: Visualizations & Diagnostics (Pygame)

Build a Pygame renderer that shows:

1. **Grid world** with colored zones
2. **Agent position** (moving dot)
3. **Heatmap overlay**: C's predicted error per cell (brighter = more curious)
4. **Real-time plots** (via matplotlib or pygame drawing):
   - Model MSE over time
   - Average curiosity reward per episode
   - Time spent in each zone type
   - External reward accumulation

### 6.1 Diagnostic Questions to Answer

- Does the agent spend more time in deterministic zones than noisy ones?
- Does curiosity speed up learning of the external task?
- What happens when the dynamic zone switches rules? (Should see a curiosity spike)
- How does `beta` affect the trade-off between exploration and goal-seeking?

---

## 7. Expected Experimental Results

Based on the 1991 paper's findings:

| Metric | Random Explorer | Curious Agent |
|--------|----------------|---------------|
| Model MSE convergence | Slow, uniform | **~10× faster**, focused on learnable zones |
| Time in noisy zones | High (wasted) | Low (correctly avoids) |
| Time in static zones | High (wasted) | Low (quickly bored) |
| External task performance | May get distracted | Often **better** due to thorough exploration |

---

## 8. Suggested Code Structure

```
curious_agent/
├── env/
│   └── grid_world.py          # Pygame environment with zones
├── agents/
│   ├── tabular_curious.py     # Phase 1: Lookup tables
│   └── dqn_curious.py         # Phase 2: Neural networks
├── models/
│   ├── world_model.py         # M network
│   ├── confidence_net.py      # C network
│   └── q_network.py           # Q network (standard + dueling)
├── utils/
│   └── replay_buffer.py
├── train_tabular.py
├── train_dqn.py
└── visualize.py               # Pygame + plotting
```

---

## 9. Quick Start Recommendation

1. **Start with Phase 1 (tabular)** — you can have a working curious agent in ~200 lines of Python
2. **Use a 5×5 grid first** — small enough to see Q-values converge
3. **Set `r_ext = 0` initially** — watch pure curiosity drive behavior
4. **Then add the external goal** — observe how curiosity helps or hinders
5. **Scale to DQN** once the mechanism is validated

---

## 10. References

- Schmidhuber, J. (1991). *Curious Model-Building Control Systems*. In Proc. IJCNN, Singapore, vol. 2, pp. 1458–1463. IEEE.
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, Cambridge.
- Pathak, D., et al. (2017). *Curiosity-driven Exploration by Self-supervised Prediction*. ICML 2017. (Modern extension)
