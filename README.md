# Neural Tree-Based Boosted Graph Neural Networks

## Project Overview

This project explores a redesigned version of **Boosted Graph Neural Networks (BGNN)** by replacing the traditional **Gradient Boosted Decision Trees (GBDT)** with **Neural Decision Trees** and experimenting with advanced **Graph Neural Network (GNN)** architectures.

The original BGNN framework combines boosting techniques with graph neural networks to improve performance on datasets where both **tabular features and graph structure** are important. However, BGNN uses **non-differentiable decision trees**, which prevents end-to-end optimization.

In this project, we propose a **fully differentiable architecture** that integrates Neural Trees with modern GNN models such as **GraphSAGE**, **GIN**, and **Transformer-based GNNs**.
