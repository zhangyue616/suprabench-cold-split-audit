# Canonical GNN run A and repeat comparison

`results/canonical_gnn_run_a/` contains the frozen primary run: 12,964 test-appearance predictions, 50 metric rows, and sanitized scientific provenance. Run B is retained separately as verification-only input. The actual run-A provenance records Python 3.11.15, Torch 2.9.0+cu128, PyG 2.8.0, CUDA 12.8, and an NVIDIA GeForce RTX 5090 (compute capability 12.0). `environments/canonical_gnn.txt` records these retained runtime facts and explicitly marks the missing package versions; it is a partial historical record, not a complete environment freeze. It is separate from the later released-job training environment.

`scientific_core.py` is an exact extraction of the graph batching, GINE/AttentiveFP training, and metric functions from the SHA-bound historical source. `data/canonical_gnn/scientific_config.json` contains the allow-listed scientific parameters and v19 determinism contract. The historical outer lifecycle and private acceptance gates are intentionally absent, so these files document the production algorithm but are not presented as a standalone bitwise rerun driver.

`results/canonical_gnn_comparison/` separately retains the 326-row prespecified A/B repeat comparison and its thresholds. That repeat receipt is distinct from the saved-prediction model-versus-null aggregation in `results/canonical_gnn_aggregation.*`.
