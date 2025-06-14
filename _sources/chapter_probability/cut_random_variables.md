---

### 3. **Simulation of Many Tosses**

* Empirically approximate the PMF via simulation.

```{code-cell} ipython3
:tags: [hide-input]

import random

def simulate_random_variable(n_trials=1000):
    def toss_two_coins():
        return random.choice(['h', 't']) + random.choice(['h', 't'])

    results = [toss_two_coins() for _ in range(n_trials)]
    X_vals = [res.count('h') for res in results]

    plt.figure(figsize=(6, 4))
    plt.hist(X_vals, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color='orange', edgecolor='black')
    plt.xticks([0, 1, 2])
    plt.xlabel('Value of X (Number of Heads)')
    plt.ylabel('Frequency')
    plt.title(f'Simulation of X over {n_trials} trials')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
simulate_random_variable()
```

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
import numpy as np

def plot_joint_pmf():
    # Possible outcomes: hh, ht, th, tt
    outcomes = ['hh', 'ht', 'th', 'tt']
    
    # X = #Heads, Y = #Tails
    values = {'hh': (2, 0), 'ht': (1, 1), 'th': (1, 1), 'tt': (0, 2)}

    joint_counts = {(x, y): 0 for x in [0, 1, 2] for y in [0, 1, 2]}
    for o in outcomes:
        pair = values[o]
        joint_counts[pair] += 1

    # Normalize to get probabilities
    joint_probs = {k: v / len(outcomes) for k, v in joint_counts.items()}

    # Convert to matrix for heatmap
    pmf_matrix = np.zeros((3, 3))
    for (x, y), p in joint_probs.items():
        pmf_matrix[2 - y, x] = p  # flip y for display

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(pmf_matrix, cmap='Blues')
    for (i, j), val in np.ndenumerate(pmf_matrix):
        if val > 0:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=12)
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['X=0', 'X=1', 'X=2'])
    ax.set_yticklabels(['Y=2', 'Y=1', 'Y=0'])  # flipped vertically
    ax.set_xlabel('X = #Heads')
    ax.set_ylabel('Y = #Tails')
    fig.colorbar(cax)
    plt.title("Joint PMF of (X, Y)")
    plt.tight_layout()
    plt.show()
plot_joint_pmf()
```

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parameters
n_trials = 30
coin_radius = 0.3
X_values = []

# Set up figure
fig, (ax_toss, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Animated Coin Tosses: Random Variable X = #Heads", fontsize=14)

# Coin toss axes setup
ax_toss.set_xlim(0, 2)
ax_toss.set_ylim(0, 1)
ax_toss.set_title("Coin Toss Outcome")
ax_toss.axis('off')

# Histogram axes setup
ax_hist.set_xlim(-0.5, 2.5)
ax_hist.set_ylim(0, n_trials // 2 + 3)
ax_hist.set_xticks([0, 1, 2])
bars = ax_hist.bar([0, 1, 2], [0, 0, 0], color='skyblue')
ax_hist.set_title("Distribution of X")
ax_hist.set_xlabel("X = number of heads")
ax_hist.set_ylabel("Frequency")

# Dynamic coin objects
coin_texts = []

def init():
    for i in range(2):
        t = ax_toss.text(0.5 + i, 0.5, "", fontsize=30, ha='center', va='center', bbox=dict(boxstyle="circle", facecolor="wheat", edgecolor="black"))
        coin_texts.append(t)
    return coin_texts + list(bars)

def update(frame):
    # Toss two coins
    outcome = [random.choice(['H', 'T']) for _ in range(2)]
    heads = outcome.count('H')
    X_values.append(heads)

    # Update coin visuals
    for i, result in enumerate(outcome):
        coin_texts[i].set_text(result)

    # Update histogram
    counts = [X_values.count(k) for k in [0, 1, 2]]
    for bar, count in zip(bars, counts):
        bar.set_height(count)

    return coin_texts + list(bars)

ani = animation.FuncAnimation(fig, update, frames=n_trials, init_func=init,
                              blit=True, interval=800, repeat=False)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
import numpy as np

def plot_joint_pmf():
    # Possible outcomes: hh, ht, th, tt
    outcomes = ['hh', 'ht', 'th', 'tt']
    
    # X = #Heads, Y = #Tails
    values = {'hh': (2, 0), 'ht': (1, 1), 'th': (1, 1), 'tt': (0, 2)}

    joint_counts = {(x, y): 0 for x in [0, 1, 2] for y in [0, 1, 2]}
    for o in outcomes:
        pair = values[o]
        joint_counts[pair] += 1

    # Normalize to get probabilities
    joint_probs = {k: v / len(outcomes) for k, v in joint_counts.items()}

    # Convert to matrix for heatmap
    pmf_matrix = np.zeros((3, 3))
    for (x, y), p in joint_probs.items():
        pmf_matrix[2 - y, x] = p  # flip y for display

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(pmf_matrix, cmap='Blues')
    for (i, j), val in np.ndenumerate(pmf_matrix):
        if val > 0:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=12)
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['X=0', 'X=1', 'X=2'])
    ax.set_yticklabels(['Y=2', 'Y=1', 'Y=0'])  # flipped vertically
    ax.set_xlabel('X = #Heads')
    ax.set_ylabel('Y = #Tails')
    fig.colorbar(cax)
    plt.title("Joint PMF of (X, Y)")
    plt.tight_layout()
    plt.show()
plot_joint_pmf()
```

```{code-cell} ipython3
:tags: [hide-input]
import random
import seaborn as sns
import pandas as pd

def simulate_joint_xy(n=500):
    data = []
    for _ in range(n):
        tosses = [random.choice(['H', 'T']) for _ in range(2)]
        x = tosses.count('H')
        y = tosses.count('T')
        data.append((x, y))

    df = pd.DataFrame(data, columns=["X", "Y"])
    plt.figure(figsize=(6, 5))
    sns.histplot(df, x="X", y="Y", bins=[3, 3], discrete=True, cbar=True)
    plt.title("Joint Distribution of X, Y from Simulation")
    plt.xlabel("X = #Heads")
    plt.ylabel("Y = #Tails")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
simulate_joint_xy()
```

```{code-cell} ipython3
:tags: [hide-input]
import networkx as nx

def plot_joint_network():
    outcomes = ['hh', 'ht', 'th', 'tt']
    values = {'hh': (2, 0), 'ht': (1, 1), 'th': (1, 1), 'tt': (0, 2)}
    G = nx.DiGraph()

    # Add outcome nodes
    for o in outcomes:
        G.add_node(o, pos=(0, outcomes.index(o)))
    
    # Add joint value nodes
    for o in outcomes:
        x, y = values[o]
        label = f'X={x}, Y={y}'
        G.add_node(label, pos=(1, outcomes.index(o)))
        G.add_edge(o, label)

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = ['lightblue' if n in outcomes else 'lightgreen' for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1600, font_size=10, edge_color='gray')
    plt.title("Mapping ω ↦ (X(ω), Y(ω))")
    plt.axis('off')
    plt.show()
plot_joint_network()
```
[^1]: More generally, vector spaces can be defined over any **field**
    $\mathbb{F}$. We take $\mathbb{F} = \mathbb{R}$ in this document to
    avoid an unnecessary diversion into abstract algebra.

[^2]: It is sometimes called the **kernel** by algebraists, but we
    eschew this terminology because the word "kernel" has another
    meaning in machine learning.

[^3]: If a normed space is complete with respect to the distance metric
    induced by its norm, we say that it is a **Banach space**.

[^4]: If an inner product space is complete with respect to the distance
    metric induced by its inner product, we say that it is a **Hilbert
    space**.

[^5]: Recall that $\mathbf{A}^{\!\top\!}\mathbf{A}$ and
    $\mathbf{A}\mathbf{A}^{\!\top\!}$ are positive semi-definite, so
    their eigenvalues are nonnegative, and thus taking square roots is
    always well-defined.

[^6]: A **neighborhood** about $\mathbf{x}$ is an open set which
    contains $\mathbf{x}$.

[^7]: $\mathcal{F}$ is required to be a $\sigma$-algebra for technical
    reasons; see [@rigorousprob].

[^8]: Note that a probability space is simply a measure space in which
    the measure of the whole space equals 1.

[^9]: This is a probabilist's version of the measure-theoretic term
    *almost everywhere*.

[^10]: In some cases it is possible to define conditional probability on
    events of probability zero, but this is significantly more technical
    so we omit it.

[^11]: The function must be measurable.

[^12]: More generally, the codomain can be any measurable space, but
    $\mathbb{R}$ is the most common case by far and sufficient for our
    purposes.

[^13]: Random variables that are continuous but not absolutely
    continuous are called **singular random variables**. We will not
    discuss them, assuming rather that all continuous random variables
    admit a density function.

[^14]: We haven't defined this yet; see the Correlation section below