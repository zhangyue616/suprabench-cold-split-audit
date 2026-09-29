# SPDX-License-Identifier: MIT
"""Historical canonical-GNN scientific core.

Extracted without algorithm changes from the SHA-bound production source.
The outer lifecycle, private review gates, and machine paths are omitted. This
module documents the model, fit, and scoring functions and is not a standalone
exact historical-run driver.
"""

import math

GateError = RuntimeError

def _graph_batch(graphs, prefix: str, device):
    import torch
    from torch_geometric.data import Batch, Data

    node = graphs[f"{prefix}_x"]
    edge_index = graphs[f"{prefix}_edge_index"]
    edge_attr = graphs[f"{prefix}_edge_attr"]
    node_ptr = graphs[f"{prefix}_node_ptr"]
    edge_ptr = graphs[f"{prefix}_edge_ptr"]
    items = []
    for index in range(len(node_ptr) - 1):
        node_start, node_end = int(node_ptr[index]), int(node_ptr[index + 1])
        edge_start, edge_end = int(edge_ptr[index]), int(edge_ptr[index + 1])
        items.append(
            Data(
                x=torch.tensor(node[node_start:node_end], dtype=torch.float),
                edge_index=torch.tensor(edge_index[:, edge_start:edge_end], dtype=torch.long),
                edge_attr=torch.tensor(edge_attr[edge_start:edge_end], dtype=torch.float),
            )
        )
    return Batch.from_data_list(items).to(device)

def _train_gnn(model_name, host_batch, guest_batch, host_index, guest_index, y, train, test, device):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch_geometric.nn import AttentiveFP, GINEConv, global_mean_pool

    class GINEEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(48, 128)
            self.convolutions = nn.ModuleList(
                [
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(128, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                        ),
                        edge_dim=6,
                    )
                    for _ in range(3)
                ]
            )
            self.normalizations = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(3)])
            self.dropout = nn.Dropout(0.1)

        def forward(self, data):
            hidden = self.linear(data.x)
            for convolution, normalization in zip(
                self.convolutions, self.normalizations, strict=True
            ):
                hidden = convolution(hidden, data.edge_index, data.edge_attr)
                hidden = normalization(hidden)
                hidden = torch.relu(hidden)
                hidden = self.dropout(hidden)
            return global_mean_pool(hidden, data.batch)

    class AttentiveFPEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AttentiveFP(
                in_channels=48,
                hidden_channels=128,
                out_channels=128,
                edge_dim=6,
                num_layers=3,
                num_timesteps=2,
                dropout=0.1,
            )

        def forward(self, data):
            return self.model(data.x, data.edge_index, data.edge_attr, data.batch)

    class PairNetwork(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )

        def forward(self, host_ids, guest_ids):
            host_embedding = self.encoder(host_batch)
            guest_embedding = self.encoder(guest_batch)
            left = host_embedding[host_ids]
            right = guest_embedding[guest_ids]
            return self.head(
                torch.cat([left, right, left * right, (left - right).abs()], dim=1)
            ).squeeze(1)

    torch.manual_seed(0)
    train = np.asarray(train)
    permutation = np.random.RandomState(0).permutation(len(train))
    validation_count = max(20, int(0.15 * len(train)))
    validation = train[permutation[:validation_count]]
    training = train[permutation[validation_count:]]
    mean = y[training].mean()
    standard_deviation = y[training].std() + 1e-6
    encoder = GINEEncoder() if model_name == "GINE" else AttentiveFPEncoder()
    model = PairNetwork(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    host_tensor = torch.tensor(host_index, device=device)
    guest_tensor = torch.tensor(guest_index, device=device)
    target = torch.tensor((y - mean) / standard_deviation, device=device)
    training_tensor = torch.tensor(training, device=device)
    validation_tensor = torch.tensor(validation, device=device)
    test_tensor = torch.tensor(test, device=device)
    validation_y = torch.tensor(y[validation], device=device)
    best = 1e9
    best_state = None
    wait = 0
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        prediction = model(host_tensor[training_tensor], guest_tensor[training_tensor])
        loss = ((prediction - target[training_tensor]) ** 2).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(host_tensor[validation_tensor], guest_tensor[validation_tensor])
                * standard_deviation
                + mean
            )
            validation_mae = torch.abs(validation_prediction - validation_y).mean().item()
        if validation_mae < best - 1e-4:
            best = validation_mae
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= 30:
            break
    if best_state is None:
        raise GateError(f"GNN produced no checkpoint: {model_name}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = (
            model(host_tensor[test_tensor], guest_tensor[test_tensor])
            * standard_deviation
            + mean
        ).cpu().numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output

def _metric_values(y_true, y_pred):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    if len(np.unique(y_true)) > 1 and np.std(y_pred) > 1e-9:
        r2 = float(r2_score(y_true, y_pred))
        r2_status = "DEFINED"
    else:
        r2 = None
        r2_status = "UNKNOWN"
    if len(np.unique(y_true)) > 1:
        candidate = float(spearmanr(y_true, y_pred).statistic)
        spearman = candidate if math.isfinite(candidate) else None
    else:
        spearman = None
    return mae, r2, spearman, r2_status, "DEFINED" if spearman is not None else "UNKNOWN"

def _metric_record(scope, split, block_id, fold_index, seed, model, y_true, y_pred):
    mae, r2, spearman, r2_status, spearman_status = _metric_values(y_true, y_pred)
    return {
        "scope": scope,
        "split": split,
        "block_id": block_id,
        "fold_index": fold_index,
        "seed": seed,
        "model": model,
        "n": str(len(y_true)),
        "mae": format(mae, ".17g"),
        "r2": "" if r2 is None else format(r2, ".17g"),
        "spearman": "" if spearman is None else format(spearman, ".17g"),
        "r2_status": r2_status,
        "spearman_status": spearman_status,
    }
