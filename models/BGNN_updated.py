import time
import numpy as np
import torch
import torch.nn as nn

from .GNN import GNNModelDGL, GATDGL
from .Base import BaseModel
from .ndt import NeuralDecisionForest
from tqdm import tqdm
from collections import defaultdict as ddict


class BGNN_NDT(BaseModel):
    def __init__(self,
                 task='regression',
                 lr=0.01,
                 hidden_dim=64,
                 dropout=0.,
                 name='gat',
                 use_leaderboard=False,
                 num_trees=5,
                 tree_depth=3):
        
        super(BaseModel, self).__init__()

        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout
        self.name = name
        self.use_leaderboard = use_leaderboard

        self.num_trees = num_trees
        self.tree_depth = tree_depth

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'BGNN_NDT'

    # ---------------- GNN ---------------- #
    def init_gnn_model(self):
        if self.use_leaderboard:
            self.model = GATDGL(
                in_feats=self.in_dim,
                n_classes=self.out_dim
            ).to(self.device)
        else:
            self.model = GNNModelDGL(
                in_dim=self.in_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                name=self.name,
                dropout=self.dropout
            ).to(self.device)

    # ---------------- NDT ---------------- #
    def init_ndt(self):
        self.ndt = NeuralDecisionForest(
            input_dim=self.raw_input_dim,
            output_dim=self.raw_input_dim,  # feature transformer
            num_trees=self.num_trees,
            depth=self.tree_depth
        ).to(self.device)

    # ---------------- Forward ---------------- #
    def forward(self, graph, x):
        x_transformed = self.ndt(x)
        return self.model(graph, x_transformed)

    # ---------------- Training ---------------- #
    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True):

        # metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3
        else:
            best_metric = [np.float('inf')] * 3

        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)

        if self.task == 'regression':
            self.out_dim = y.shape[1]
        else:
            self.out_dim = len(set(y.iloc[test_mask, 0]))

        # -------- Feature preprocessing -------- #
        encoded_X = X.copy()
        if len(cat_features):
            encoded_X = self.encode_cat_features(encoded_X, y, cat_features,
                                                 train_mask, val_mask, test_mask)

        if normalize_features:
            encoded_X = self.normalize_features(encoded_X,
                                                train_mask, val_mask, test_mask)

        if replace_na:
            encoded_X = self.replace_na(encoded_X, train_mask)

        # convert to torch
        x = torch.from_numpy(encoded_X.to_numpy()).float().to(self.device)
        self.raw_input_dim = x.shape[1]

        y, = self.pandas_to_torch(y)
        self.y = y

        # graph
        graph = self.networkx_to_torch(networkx_graph)
        self.graph = graph

        # init models
        self.in_dim = self.raw_input_dim
        self.init_gnn_model()
        self.init_ndt()

        # optimizer (joint!)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.ndt.parameters()),
            lr=self.learning_rate
        )

        # ---------------- Training Loop ---------------- #
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start = time.time()

            optimizer.zero_grad()

            out = self.forward(graph, x)

            loss = self.train_and_evaluate(
                (graph, out), y,
                train_mask, val_mask, test_mask,
                optimizer=None,  # we handle manually
                metrics=metrics,
                iter_per_epoch=1
            )

            loss.backward()
            optimizer.step()

            self.log_epoch(
                pbar, metrics, epoch, loss,
                time.time() - start,
                logging_epochs,
                metric_name=metric_name
            )

            # early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(
                    metrics, epoch,
                    best_metric,
                    best_val_epoch,
                    epochs_since_last_best_metric,
                    metric_name,
                    lower_better=(metric_name not in ['r2', 'accuracy'])
                )

            if patience and epochs_since_last_best_metric > patience:
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(
            metric_name, best_val_epoch, *best_metric))

        return metrics

    def predict(self, graph, X, y, test_mask):
        x = torch.from_numpy(X.to_numpy()).float().to(self.device)
        out = self.forward(graph, x)
        return self.evaluate_model((graph, out), y, test_mask)
