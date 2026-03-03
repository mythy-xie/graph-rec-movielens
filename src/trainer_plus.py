import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

from src.data_loader import MovieLensDataLoader
from src.utils import compute_rmse, compute_ranking_metrics
from src.model import MovieLensSAGE, MovieLensGCN, MovieLensGGNN, LinkPredictor


def setup_logger(args: argparse.Namespace) -> logging.Logger:
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{args.dataset}_{args.model}_lr{args.lr}_cs{args.cold_start_ratio}_{timestamp}.log"

    logger = logging.getLogger("TrainerPlus")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"The log file has been saved to: {log_filename}")
    return logger


class ModelTrainerPlus:

    def __init__(self, data: HeteroData, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger

        if args.device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(args.device)

        self.logger.info(f"Initialize TrainerPlus, using the device: {self.device} | Model: {args.model}")

        num_users = data['user'].num_nodes
        num_cs_users = int(num_users * args.cold_start_ratio)

        all_user_indices = torch.randperm(num_users)
        cs_user_indices = all_user_indices[:num_cs_users]
        warm_user_indices = all_user_indices[num_cs_users:]

        edge_index = data['user', 'rates', 'movie'].edge_index
        is_cs_edge = torch.isin(edge_index[0], cs_user_indices)

        self.test_data = data.clone()
        self.test_data['user', 'rates', 'movie'].edge_label_index = edge_index[:, is_cs_edge]
        self.test_data['user', 'rates', 'movie'].edge_label = data['user', 'rates', 'movie'].edge_label[is_cs_edge]
        self.test_data['user', 'rates', 'movie'].edge_index = edge_index[:, ~is_cs_edge]

        train_base_data = data.clone()
        train_base_data['user', 'rates', 'movie'].edge_index = edge_index[:, ~is_cs_edge]
        train_base_data['user', 'rates', 'movie'].edge_label = data['user', 'rates', 'movie'].edge_label[~is_cs_edge]

        transform = T.RandomLinkSplit(
            num_val=0.1, num_test=0.0,
            neg_sampling_ratio=0.0,
            edge_types=[('user', 'rates', 'movie')],
            rev_edge_types=[('movie', 'rated_by', 'user')],
        )
        self.train_data, self.val_data, _ = transform(train_base_data)

        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)
        self.test_data = self.test_data.to(self.device)

        self.logger.info(f"Explicit cold start splitting complete | Cold-Start Ratio: {args.cold_start_ratio}")
        self.logger.info(f"   - Warm User: {len(warm_user_indices)} | Cold User: {len(cs_user_indices)}")
        self.logger.info(f"   - Inductive evaluation of edge count (Test): {is_cs_edge.sum().item()}")

        self.logger.info(f"Building {args.model} architecture...")
        if args.model == 'SAGE':
            self.model = MovieLensSAGE(hidden_channels=args.hidden_channels, out_channels=16).to(self.device)
        elif args.model == 'GCN':
            self.model = MovieLensGCN(hidden_channels=args.hidden_channels, out_channels=16).to(self.device)
        elif args.model == 'GGNN':
            self.model = MovieLensGGNN(hidden_channels=args.hidden_channels, out_channels=16).to(self.device)
        else:
            self.logger.error(f"Unsupported model type requested: {args.model}")
            raise ValueError(f"Unsupported model type: {args.model}")

        self.logger.info(f"Successfully instantiated {args.model} and moved to {self.device}.")

        self.predictor = LinkPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            lr=args.lr, weight_decay=1e-4
        )

        self.best_val_rmse = float('inf')
        self.save_dir = project_root / 'checkpoints'
        self.save_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.save_dir / f"best_model_{args.dataset}_{args.model}.pth"

    def train_epoch(self) -> float:
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        node_embeddings = self.model(self.train_data.x_dict, self.train_data.edge_index_dict)
        edge_label_index = self.train_data['user', 'rates', 'movie'].edge_label_index
        edge_label = self.train_data['user', 'rates', 'movie'].edge_label

        preds = self.predictor(node_embeddings['user'], node_embeddings['movie'], edge_label_index)
        loss = F.mse_loss(preds, edge_label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data_split: HeteroData, k: int = 10) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        self.predictor.eval()

        node_embeddings = self.model(data_split.x_dict, data_split.edge_index_dict)
        edge_label_index = data_split['user', 'rates', 'movie'].edge_label_index
        edge_label = data_split['user', 'rates', 'movie'].edge_label

        preds = self.predictor(node_embeddings['user'], node_embeddings['movie'], edge_label_index)

        rmse = compute_rmse(preds, edge_label)
        ranking_metrics = compute_ranking_metrics(
            edge_label_index[0], edge_label_index[1], preds, edge_label, k=k
        )
        return rmse, ranking_metrics

    def run(self):
        self.logger.info(f"Start training, total Epoch: {self.args.epochs}")

        total_train_time = 0.0

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            loss = self.train_epoch()
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time

            val_rmse, val_metrics = self.evaluate(self.val_data)

            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    f"Epoch {epoch:03d} [{epoch_time:.3f}s] | Loss: {loss:.4f} | Val RMSE: {val_rmse:.4f} | "
                    f"Val Recall@10: {val_metrics['Recall@10']:.4f} | Val NDCG@10: {val_metrics['NDCG@10']:.4f}")

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                torch.save({
                    'model_state': self.model.state_dict(),
                    'predictor_state': self.predictor.state_dict(),
                }, self.checkpoint_path)

        self.logger.info(
            f"Total training time: {total_train_time:.2f}s | Per epoch: {(total_train_time / self.args.epochs):.4f}s")
        self.logger.info("Training complete! Now, begin evaluating inductive generalization ability on the Cold Start Test Set...")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.predictor.load_state_dict(checkpoint['predictor_state'])

        infer_start_time = time.time()
        test_rmse, test_metrics = self.evaluate(self.test_data)
        infer_time = time.time() - infer_start_time

        self.logger.info(f"[Cold start generalization ability assessment (Inductive Generalization)]")
        self.logger.info(f"   - Test RMSE: {test_rmse:.4f}")
        self.logger.info(f"   - Test Recall@10: {test_metrics['Recall@10']:.4f}")
        self.logger.info(f"   - Test NDCG@10: {test_metrics['NDCG@10']:.4f}")
        self.logger.info(f"Test set inference time: {infer_time:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens Graph Recommender System - Advanced Trainer")
    parser.add_argument('--dataset', type=str, choices=['100k', '1m'], default='100k', help='Dataset version')
    parser.add_argument('--model', type=str, choices=['SAGE', 'GCN', 'GGNN'], default='SAGE', help='GNN Model selection')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden layer dimensions')
    parser.add_argument('--cold_start_ratio', type=float, default=0.1, help='Cold start (Inductive) user percentage')
    parser.add_argument('--device', type=str, default='auto', help='computing devices (auto/cpu/cuda/mps)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"MovieLens GNN Experiments | Dataset: {args.dataset} | Model: {args.model}")
    print("=" * 60 + "\n")
    logger = setup_logger(args)

    data_dir = project_root / "data" / f"ml-{args.dataset}"
    loader = MovieLensDataLoader(data_dir=data_dir, version=args.dataset)
    loader.load_raw_data()
    graph_data = loader.build_graph()

    trainer = ModelTrainerPlus(data=graph_data, args=args, logger=logger)
    trainer.run()
