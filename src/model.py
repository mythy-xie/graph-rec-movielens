import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from typing import Dict


class MovieLensSAGE(torch.nn.Module):

    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        self.lin_dict = torch.nn.ModuleDict({
            'user': Linear(-1, hidden_channels),
            'movie': Linear(-1, hidden_channels)
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'rates', 'movie'): SAGEConv(hidden_channels, hidden_channels),
                ('movie', 'rated_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.out_lin_dict = torch.nn.ModuleDict({
            'user': Linear(hidden_channels, out_channels),
            'movie': Linear(hidden_channels, out_channels)
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        x_dict = {
            node_type: F.leaky_relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.dropout(F.leaky_relu(x), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        out_dict = {
            node_type: self.out_lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        return out_dict


class LinkPredictor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_user: torch.Tensor, x_movie: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        user_node_indices = edge_label_index[0]
        movie_node_indices = edge_label_index[1]

        user_embeddings = x_user[user_node_indices]
        movie_embeddings = x_movie[movie_node_indices]

        predictions = (user_embeddings * movie_embeddings).sum(dim=-1)

        return predictions


if __name__ == "__main__":
    print("Model architecture initialization testing is underway...")

    hidden_dim = 32
    out_dim = 16

    try:
        model = MovieLensSAGE(hidden_channels=hidden_dim, out_channels=out_dim)
        predictor = LinkPredictor()

        dummy_x_dict = {
            'user': torch.randn(10, 100),
            'movie': torch.randn(20, 19)
        }
        dummy_edge_index_dict = {
            ('user', 'rates', 'movie'): torch.randint(0, 10, (2, 50)),
            ('movie', 'rated_by', 'user'): torch.randint(0, 10, (2, 50))
        }

        embeddings = model(dummy_x_dict, dummy_edge_index_dict)
        assert embeddings['user'].shape == (10, out_dim), "Error in User Embedding Dimension"
        assert embeddings['movie'].shape == (20, out_dim), "Error in Movie Embedding Dimension"
        print(f"Encoder (GraphSAGE) Forward propagation successful! Output dimension: {out_dim}")

        dummy_edge_label_index = torch.tensor([[0, 1, 2, 3, 4], [0, 5, 10, 15, 19]])
        preds = predictor(embeddings['user'], embeddings['movie'], dummy_edge_label_index)
        assert preds.shape == (5,), "Error in predicting output dimension"
        print(f"Decoder (LinkPredictor) Forward propagation successful!")

    except Exception as e:
        print(f"Model test failed: {e}")