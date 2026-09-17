# Semi-Decentralized Federated Learning

A semi-decentralized Federated Learning (FL) system for classifying premature deaths caused by Non-Communicable Diseases (NCDs), using tabular hospital data spread across multiple nodes (e.g. Raspberry Pi devices), with no central server holding the raw data.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Your PC (DB)    │         │  Raspberry Pi 1 │
│   PseudoDB       │◄───────►│   Hospital 1    │
│ 172.23.211.109   │         │   data1.csv     │
└────────┬─────────┘         └─────────────────┘
         │
         │                  ┌─────────────────┐
         ├─────────────────►│  Raspberry Pi 2 │
         │                  │   Hospital 2    │
         │                  │   data2.csv     │
         │                  └─────────────────┘
         │
         │                  ┌─────────────────┐
         ├─────────────────►│  Raspberry Pi 3 │
         │                  │   Hospital 3    │
         │                  │   data3.csv     │
         │                  └─────────────────┘
         │
         │                  ┌─────────────────┐
         └─────────────────►│  Raspberry Pi 4 │
                            │   Hospital 4    │
                            │   data4.csv     │
                            └─────────────────┘
```

The system has two independent deployment units:

- **[deploy_db_server/](deploy_db_server/)** — a PseudoDB coordination server (SQLite-backed) that runs on a central machine. It never sees raw hospital data, only model updates and node registration/discovery.
- **[deploy_node/](deploy_node/)** — the FL node, deployed once per hospital/device. Each node trains locally on its own CSV and starts out as an `agent`. Nodes dynamically elect one of themselves as the round's **aggregator**; on promotion, a `role_supervisor` process swaps the node from the client (`tabular_engine`) to the aggregator server (`server_th`) in place, and demotes it back afterwards. This is what makes the setup "semi-decentralized": there's a lightweight discovery service, but aggregation itself rotates among the nodes instead of living on a fixed server.

## Training flow

1. All nodes connect to the PseudoDB.
2. An initial aggregator is elected (by connection order).
3. Each node trains locally on its own data.
4. Nodes send gradients/weights to the current aggregator.
5. The aggregator averages the models (FedAvg).
6. The global model is redistributed to all nodes.
7. Repeat until convergence.
8. Every N rounds, a new aggregator is elected.

## Model

Tabular MLP classifier, 21 input features → probability of premature NCD death:

```
Input (21 features) → fc1:120 + ReLU → fc2:84 + ReLU → fc3:1 + Sigmoid → P(premature NCD death)
```

- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr=0.001)
- Local epochs: 5, batch size: 32, train/test split: 80/20

Target column: `is_premature_ncd` (1 = death before age 70 from an NCD, 0 = other cause).

## Getting started

### 1. Start the DB server (on your PC)

```bash
cd deploy_db_server
pip install -r requirements.txt
# edit setups/config_db.json with your IP/port
./scripts/start.sh
```

### 2. Start each node (on each Raspberry Pi / hospital device)

```bash
cd deploy_node
pip install -r requirements.txt
# copy the hospital's CSV to data/data.csv
# edit setups/config_agent.json: set device_ip to this machine's IP, db_ip/db_port to the server's
./scripts/start.sh
```

`start.sh` checks dependencies, validates config and data files, clears stale processes on the FL ports, and launches the node via `role_supervisor`.

See [deploy_db_server/README.md](deploy_db_server/README.md) and [deploy_node/README_DESPLIEGUE.md](deploy_node/README_DESPLIEGUE.md) for full configuration details, the dataset schema, and troubleshooting.

## Notes

- Raw data never leaves a node — only model weights/gradients are exchanged.
- The preprocessor (`artifacts/preprocessor_global.joblib`) must be identical across all nodes.
- The system tolerates temporary node disconnections.
