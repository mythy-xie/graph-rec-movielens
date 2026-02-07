import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieLensDataLoader:

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.movies_path = self.data_dir / 'u.item'
        self.ratings_path = self.data_dir / 'u.data'
        self.users_path = self.data_dir / 'u.user'

        self.user_id_map: Dict[int, int] = {}
        self.movie_id_map: Dict[int, int] = {}

        self.df_movies: Optional[pd.DataFrame] = None
        self.df_ratings: Optional[pd.DataFrame] = None
        self.df_users: Optional[pd.DataFrame] = None

    def load_raw_data(self) -> None:
        logger.info("Loading Data...")

        movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + \
                     ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        self.df_movies = pd.read_csv(
            self.movies_path, sep='|', names=movie_cols, encoding='ISO-8859-1', usecols=range(24)
        )

        self.df_ratings = pd.read_csv(
            self.ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        self.df_users = pd.read_csv(
            self.users_path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )

        logger.info(
            f"Data Loaded: Movies={len(self.df_movies)}, Users={len(self.df_users)}, Ratings={len(self.df_ratings)}")

    def _create_id_mappings(self) -> None:
        unique_users = self.df_users['user_id'].unique()
        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}

        unique_movies = self.df_movies['movie_id'].unique()
        self.movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}

        logger.info("ID Mapping Completed.")

    def process_node_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        genre_cols = self.df_movies.columns[5:]
        movie_features = torch.from_numpy(self.df_movies[genre_cols].values).to(torch.float)

        num_users = len(self.user_id_map)
        user_features = torch.eye(num_users)

        return movie_features, user_features

    def build_graph(self) -> HeteroData:
        if self.df_ratings is None:
            self.load_raw_data()

        self._create_id_mappings()

        data = HeteroData()

        movie_x, user_x = self.process_node_features()

        data['user'].num_nodes = len(self.user_id_map)
        data['user'].x = user_x

        data['movie'].num_nodes = len(self.movie_id_map)
        data['movie'].x = movie_x

        src = [self.user_id_map[uid] for uid in self.df_ratings['user_id']]
        dst = [self.movie_id_map[mid] for uid, mid in zip(self.df_ratings['user_id'], self.df_ratings['movie_id'])]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(self.df_ratings['rating'].values, dtype=torch.float)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = edge_attr

        data['movie', 'rated_by', 'user'].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        logger.info(f"Graph Constructed.: {data}")
        return data

    def perform_eda(self, save_dir: str | Path = 'plots') -> None:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(8, 5))

        sns.countplot(x='rating', data=self.df_ratings, palette='viridis', hue='rating', legend=False)
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating Score')
        plt.ylabel('Count')
        plt.savefig(save_path / 'rating_distribution.png')
        plt.close()
        logger.info(f"The rating distribution chart has been saved to: {save_path / 'rating_distribution.png'}")

        # Plot 2: 长尾效应 (保持不变)
        movie_counts = self.df_ratings.groupby('movie_id').size().sort_values(ascending=False).reset_index(name='count')
        plt.figure(figsize=(10, 6))
        plt.plot(movie_counts.index, movie_counts['count'], color='blue', alpha=0.7)
        plt.fill_between(movie_counts.index, movie_counts['count'], color='blue', alpha=0.1)
        plt.title('Long-tail Distribution (Ratings per Movie)')
        plt.xlabel('Movie Index (Sorted by popularity)')
        plt.ylabel('Number of Ratings')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(save_path / 'long_tail_movies.png')
        plt.close()
        logger.info(f"The long-tail distribution map has been saved to: {save_path / 'long_tail_movies.png'}")

        genre_sums = self.df_movies.iloc[:, 5:].sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 8))

        sns.barplot(x=genre_sums.values, y=genre_sums.index, palette='magma', hue=genre_sums.index, legend=False)
        plt.title('Most Popular Movie Genres')
        plt.xlabel('Number of Movies')
        plt.tight_layout()
        plt.savefig(save_path / 'genre_popularity.png')
        plt.close()
        logger.info(f"The category distribution chart has been saved to: {save_path / 'genre_popularity.png'}")


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    data_path = project_root / "data" / "ml-100k"

    print(f"DEBUG: Project root directory detected as: {project_root}")
    print(f"DEBUG: Attempting to read data path.: {data_path}")

    if not data_path.exists():
        print(f"\n WRONG: Folder {data_path} Not Found！")
        print("Please check if the `data` folder is located in the project's root directory and that its name is correct.")
        exit(1)

    print("Path check successful, starting to load...")

    loader = MovieLensDataLoader(data_dir=data_path)
    loader.load_raw_data()

    loader.perform_eda(save_dir=project_root / 'plots')

    graph = loader.build_graph()

    print("\n=== Graph Summary ===")
    print(graph)
    print(f"Edge types: {graph.edge_types}")
