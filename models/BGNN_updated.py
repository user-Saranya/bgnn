import time
import numpy as np
import torch
import torch.nn.functional as F

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
                 name='sage',
                 use_leaderboard=False,
                 num_trees=5,
                 tree_depth=3):

        super().__init__()

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
            output_dim=self.raw_input_dim,   # feature transformation
            num_trees=self.num_trees,
            depth=self.tree_depth
        ).to(self.device)

    # ---------------- Forward ---------------- #
    def forward(self, graph, x):
        x_transformed = self.ndt(x)
        out = self.model(graph, x_transformed)
        return out

    # ---------------- Training ---------------- #
    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True):

        # -------- Metrics setup -------- #
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3
        else:
            best_metric = [np.float('inf')] * 3

        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)

        if cat_features is None:
            cat_features = []

        # -------- Output dimension -------- #
        if self.task == 'regression':
            self.out_dim = y.shape[1]
        else:
            self.out_dim = len(set(y.iloc[test_mask, 0]))

        # -------- Feature preprocessing -------- #
        encoded_X = X.copy()

        if len(cat_features):
            encoded_X = self.encode_cat_features(
                encoded_X, y, cat_features,
                train_mask, val_mask, test_mask
            )

        if normalize_features:
            encoded_X = self.normalize_features(
                encoded_X, train_mask, val_mask, test_mask
            )

        if replace_na:
            encoded_X = self.replace_na(encoded_X, train_mask)

        # -------- Convert to torch -------- #
        x = torch.from_numpy(encoded_X.to_numpy()).float().to(self.device)
        self.raw_input_dim = x.shape[1]
        self.in_dim = self.raw_input_dim

        y, = self.pandas_to_torch(y)
        self.y = y

        # -------- Graph -------- #
        graph = self.networkx_to_torch(networkx_graph)
        self.graph = graph

        # -------- Init models -------- #
        self.init_gnn_model()
        self.init_ndt()

        # -------- Optimizer (joint!) -------- #
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.ndt.parameters()),
            lr=self.learning_rate
        )

        # ---------------- Training Loop ---------------- #
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start = time.time()

            self.model.train()
            self.ndt.train()

            # forward
            out = self.forward(graph, x)

            # loss
            if self.task == 'regression':
                loss = torch.sqrt(F.mse_loss(out[train_mask], y[train_mask]))
            else:
                loss = F.cross_entropy(out[train_mask], y[train_mask].long())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------- Evaluation -------- #
            self.model.eval()
            self.ndt.eval()

            with torch.no_grad():
                out = self.forward(graph, x)

                train_res = self.evaluate_model(out, y, train_mask)
                val_res = self.evaluate_model(out, y, val_mask)
                test_res = self.evaluate_model(out, y, test_mask)

            for key in train_res:
                metrics[key].append((
                    train_res[key].item(),
                    val_res[key].item(),
                    test_res[key].item()
                ))

            # -------- Logging -------- #
            self.log_epoch(
                pbar, metrics, epoch, loss,
                time.time() - start,
                logging_epochs,
                metric_name=metric_name
            )

            # -------- Early stopping -------- #
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

    # ---------------- Prediction ---------------- #
    def predict(self, graph, X, y, test_mask):
        x = torch.from_numpy(X.to_numpy()).float().to(self.device)

        self.model.eval()
        self.ndt.eval()

        with torch.no_grad():
            out = self.forward(graph, x)

        return self.evaluate_model(out, y, test_mask)
