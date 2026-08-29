## Overview

This project implements two graph kernels **from scratch** for comparing and classifying labeled graphs:

* **Weisfeiler-Lehman (WL) Graph Kernel**
* **Generalized Weisfeiler-Lehman (GWL) Graph Kernel**

The main goal is to explore how the **structure and labels of a graph can be transformed into useful numerical representations**, which can then be used with standard machine-learning algorithms for tasks such as graph classification.

### Why Graph Kernels?

Graphs are a natural way to represent structured data. For example, in a social-media propagation tree, each node can represent a user and each edge can represent an interaction or propagation event.

Unlike traditional data such as images or tables, graphs do not have a fixed-size structure. Different graphs can contain different numbers of nodes and edges, making them difficult to use directly with conventional machine-learning models.

Graph kernels provide a way to overcome this problem.

Instead of feeding the entire graph directly into a machine-learning model, we extract information about its **local structure and node labels** and use this information to measure how similar two graphs are.

### How Does the Weisfeiler-Lehman Kernel Work?

The Weisfeiler-Lehman approach works by iteratively updating the label of each node based on:

1. Its current label.
2. The labels of its neighboring nodes.

At each iteration, the neighborhood information is combined and assigned a new label. Repeating this process allows the algorithm to capture increasingly larger structural patterns within the graph.

In simple terms:

```text
Initial node labels
        ↓
Collect neighboring labels
        ↓
Create new node labels
        ↓
Repeat for several iterations
        ↓
Extract subtree patterns
        ↓
Compute graph similarity
```

The extracted subtree patterns are counted and used to compute a kernel matrix, where each entry represents the similarity between a pair of graphs based on the subtree patterns they share.

### Generalized Weisfeiler-Lehman Kernel

The Generalized Weisfeiler-Lehman (GWL) kernel extends the standard WL approach by working with **unfolding trees** that represent the local structure around each node.

Instead of only assigning new discrete labels to nodes at each WL iteration, the generalized approach analyzes the resulting unfolding-tree structures and **groups them into buckets based on their structural characteristics**.

In simple terms:

```text
Graph
  ↓
Construct unfolding trees
  ↓
Analyze their structural patterns
  ↓
Group similar unfolding trees into buckets
  ↓
Use the resulting patterns/buckets as graph features
  ↓
Compute graph similarities
```

This allows the kernel to capture structural similarities between graphs at the level of their local unfolding-tree patterns.

The main difference between the two approaches in this project is therefore how the local subtree structures are represented and compared: the standard WL kernel relies on iterative label refinement, while the generalized version uses the structure of unfolding trees and their grouping into buckets to obtain a more flexible representation.


### Dataset

The project can be applied to datasets containing labeled graphs and propagation structures. In particular, the implementation is designed to work with **Twitter15/Twitter16-style rumor propagation data**, where each conversation can be represented as a propagation tree.

In this setting:

* **Nodes** represent users participating in the propagation.
* **Edges** represent propagation relationships.
* **Node labels/features** describe information associated with the nodes.
* **The graph label** represents the class of the complete propagation tree.

This allows the problem of rumor detection to be formulated as a **graph classification problem**.

### Project Goal

The purpose of this repository is not only to use an existing graph-kernel library, but to provide a **from-scratch implementation** of the main ideas behind WL-based graph kernels.

This makes it possible to:

* Understand how WL graph kernels work internally.
* Experiment with different graph representations.
* Compare standard and generalized WL approaches.
* Study how graph structure influences classification.
* Use the resulting kernel representations with conventional machine-learning algorithms.

Ultimately, the project aims to provide a clear and experimental framework for understanding **how graph structure can be transformed into meaningful features for machine learning**.
