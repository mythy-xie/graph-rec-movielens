import math
from collections import defaultdict
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.view(-1)
    targets = targets.view(-1)

    mse = F.mse_loss(preds, targets)
    rmse = torch.sqrt(mse)
    return rmse.item()


def compute_ranking_metrics(
        user_indices: torch.Tensor,
        movie_indices: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
        threshold: float = 4.0
) -> Dict[str, float]:
    users = user_indices.tolist()
    movies = movie_indices.tolist()
    predictions = preds.tolist()
    ground_truths = targets.tolist()

    user_interactions = defaultdict(list)
    for u, m, p, t in zip(users, movies, predictions, ground_truths):
        user_interactions[u].append((m, p, t))

    recalls = []
    ndcgs = []

    for u, items in user_interactions.items():
        true_relevant_items = [item for item in items if item[2] >= threshold]
        num_true_relevant = len(true_relevant_items)

        if num_true_relevant == 0:
            continue

        items.sort(key=lambda x: x[1], reverse=True)
        top_k_items = items[:k]

        hits = sum(1 for item in top_k_items if item[2] >= threshold)
        recall = hits / num_true_relevant
        recalls.append(recall)

        dcg = 0.0
        for rank, item in enumerate(top_k_items):
            if item[2] >= threshold:
                dcg += 1.0 / math.log2(rank + 2)

        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, num_true_relevant)))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0
    }


if __name__ == "__main__":
    print("Testing the Metrics Utils...")

    dummy_users = torch.tensor([0, 0, 0, 1, 1])
    dummy_movies = torch.tensor([10, 11, 12, 20, 21])

    dummy_preds = torch.tensor([4.8, 3.2, 2.1, 4.5, 3.0])

    dummy_targets = torch.tensor([5.0, 4.0, 1.0, 2.0, 5.0])

    rmse_val = compute_rmse(dummy_preds, dummy_targets)
    print(f"RMSE calculation results: {rmse_val:.4f}")

    metrics = compute_ranking_metrics(
        dummy_users, dummy_movies, dummy_preds, dummy_targets, k=2, threshold=4.0
    )

    print(f"Ranking Metrics calculation results: {metrics}")