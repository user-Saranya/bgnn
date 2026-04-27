import time
import numpy as np
import torch
import torch.nn.functional as F

from .Base import BaseModel
from .ndt import NeuralDecisionForest
from .graph_transformer import GraphTransformer
from tqdm import tqdm
from collections import defaultdict as ddict


class BGNN_Transformer(BaseModel):
    def __init__(self,
                 task='regression',
                 lr=0.01,
                 hidden_dim=64,
                 dropout=0.,
                 num_heads=4,
                 num_layers=2,
                 num_trees=5,
                 tree_depth=3,
                 alpha=0.2):

        super().__init__()

        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.num_trees = num_trees
        self.tree_depth = tree_depth

        self.alpha = alpha  # residual scaling

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'BGNN_Transformer'

    # ---------------- NDT ---------------- #
    def init_ndt(self):
        self.ndt = NeuralDecisionForest(
            input_dim=self.raw_input_dim,
            output_dim=self.raw_input_dim,
            num_trees=self.num_trees,
            depth=self.tree_depth
        ).to(self.device)

    # ---------------- Transformer GNN ---------------- #
    def init_gnn_model(self):
        self.model = GraphTransformer(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

    # ---------------- Forward ---------------- #
    def forward(self, graph, x):
        x_ndt = self.ndt(x)

        # residual-style boosting
        x_refined = x + self.alpha * x_ndt

        # concatenate features
        x_combined = torch.cat([x_refined, x_ndt], dim=1)

        # optional stabilization
        x_combined = F.dropout(x_combined, p=self.dropout, training=self.training)

        out = self.model(graph, x_combined)
        return out

    # ---------------- Preprocessing ---------------- #
    def preprocess(self, X, y, train_mask, val_mask, test_mask, cat_features,
                   normalize_features, replace_na):

        encoded_X = X.copy()

        if cat_features is None:
            cat_features = []

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

        return encoded_X

    # ---------------- Training ---------------- #
    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True):

        # -------- Metrics -------- #
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3
        else:
            best_metric = [np.float('inf')] * 3

        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)

        # -------- Output dim -------- #
        if self.task == 'regression':
            self.out_dim = y.shape[1]
        else:
            self.out_dim = len(set(y.iloc[test_mask, 0]))

        # -------- Preprocess -------- #
        encoded_X = self.preprocess(
            X, y, train_mask, val_mask, test_mask,
            cat_features, normalize_features, replace_na
        )

        # store for prediction consistency
        self.processed_columns = encoded_X.columns

        # -------- Torch conversion -------- #
        x = torch.from_numpy(encoded_X.to_numpy()).float().to(self.device)
        self.raw_input_dim = x.shape[1]

        # concatenation doubles input dim
        self.in_dim = 2 * self.raw_input_dim

        y, = self.pandas_to_torch(y)
        self.y = y

        # -------- Graph -------- #
        graph = self.networkx_to_torch(networkx_graph)
        self.graph = graph

        # -------- Init models -------- #
        self.init_ndt()
        self.init_gnn_model()

        # -------- Optimizer -------- #
        optimizer = torch.optim.Adam(
            list(self.ndt.parameters()) + list(self.model.parameters()),
            lr=self.learning_rate
        )

        # ---------------- Training Loop ---------------- #
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start = time.time()

            self.model.train()
            self.ndt.train()

            out = self.forward(graph, x).squeeze(-1)

            # -------- Loss -------- #
            if self.task == 'regression':
                loss = F.mse_loss(out[train_mask], y[train_mask])
            else:
                loss = F.cross_entropy(out[train_mask], y[train_mask].long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------- Evaluation -------- #
            self.model.eval()
            self.ndt.eval()

            with torch.no_grad():
                out = self.forward(graph, x).squeeze(-1)

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
        encoded_X = X.copy()

        # ensure same column order
        encoded_X = encoded_X[self.processed_columns]

        x = torch.from_numpy(encoded_X.to_numpy()).float().to(self.device)

        self.model.eval()
        self.ndt.eval()

        with torch.no_grad():
            out = self.forward(graph, x).squeeze(-1)

        return self.evaluate_model(out, y, test_mask)
