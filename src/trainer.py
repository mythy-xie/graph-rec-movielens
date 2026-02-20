import logging
import sys
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
from src.model import MovieLensSAGE, LinkPredictor
from src.utils import compute_rmse, compute_ranking_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, data: HeteroData, hidden_channels: int = 32, lr: float = 0.01, device: str = 'auto'):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initialize the Trainer and use the device: {self.device}")

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=0.0,
            edge_types=[('user', 'rates', 'movie')],
            rev_edge_types=[('movie', 'rated_by', 'user')],
        )

        train_data, val_data, test_data = transform(data)

        self.train_data = train_data.to(self.device)
        self.val_data = val_data.to(self.device)
        self.test_data = test_data.to(self.device)

        logger.info("Graph cutting completed (Train/Val/Test).")

        self.model = MovieLensSAGE(hidden_channels=hidden_channels, out_channels=16).to(self.device)
        self.predictor = LinkPredictor().to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            lr=lr,
            weight_decay=1e-4
        )

        self.best_val_rmse = float('inf')
        self.save_dir = project_root / 'checkpoints'
        self.save_dir.mkdir(exist_ok=True)

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

        users = edge_label_index[0]
        movies = edge_label_index[1]
        ranking_metrics = compute_ranking_metrics(users, movies, preds, edge_label, k=k)

        return rmse, ranking_metrics

    def run(self, epochs: int = 50):
        logger.info(f"Start training, total Epoch: {epochs}")

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            val_rmse, val_metrics = self.evaluate(self.val_data)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d} | Train Loss (MSE): {loss:.4f} | Val RMSE: {val_rmse:.4f} | "
                            f"Val Recall@10: {val_metrics['Recall@10']:.4f} | Val NDCG@10: {val_metrics['NDCG@10']:.4f}")

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.save_checkpoint('best_model.pth')

        logger.info("Training complete! Begin final evaluation on the unseen Test Set....")

        self.load_checkpoint('best_model.pth')
        test_rmse, test_metrics = self.evaluate(self.test_data)
        logger.info(f"Final test set performance | Test RMSE: {test_rmse:.4f} | "
                    f"Test Recall@10: {test_metrics['Recall@10']:.4f} | Test NDCG@10: {test_metrics['NDCG@10']:.4f}")

    def save_checkpoint(self, filename: str):
        path = self.save_dir / filename
        torch.save({
            'model_state': self.model.state_dict(),
            'predictor_state': self.predictor.state_dict(),
        }, path)

    def load_checkpoint(self, filename: str):
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.predictor.load_state_dict(checkpoint['predictor_state'])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MovieLens Graph Recommender System Start")
    print("=" * 50 + "\n")

    data_dir = project_root / "data" / "ml-100k"
    loader = MovieLensDataLoader(data_dir=data_dir, version='100k')
    loader.load_raw_data()
    graph_data = loader.build_graph()

    trainer = ModelTrainer(data=graph_data, hidden_channels=32, lr=0.01)
    trainer.run(epochs=100)