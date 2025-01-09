import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from os.path import join
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.stats import truncnorm


# Matrix Generator Class
class MatrixGenerator:
    @staticmethod
    def get_matrix(
        m=None,
        n=None,
        k=10,
        random_state=None,
        missing_fraction=0.1,
        noise_level=0.1,
        with_graphs=False,
        dataset_name=None,
    ):
        """
        Returns a matrix object based on the given parameters.
        If a dataset name is provided, it returns a DatasetMatrix.
        Otherwise, it returns a RandomMatrix.
        """
        if dataset_name:
            # Return a DatasetMatrix if dataset_name is provided
            dm = DatasetMatrix(
                dataset_name=dataset_name,
                rank=k,
                random_state=random_state,
            )
            return dm.generate()
        else:
            # Ensure dimensions m and n are provided for RandomMatrix
            if m is None or n is None:
                raise ValueError(
                    "For RandomMatrix, both 'm' (rows) and 'n' (columns) must be specified."
                )
            # Return a RandomMatrix otherwise
            rm = RandomMatrix(
                m=m,
                n=n,
                k=k,
                random_state=random_state,
                missing_fraction=missing_fraction,
                noise_level=noise_level,
            )
            if not with_graphs:
                return rm.generate()
            else:
                A_r = rm.build_A_matrices_from_laplacians(p=2)
                return rm.generate_with_graphs(A_r)


# Successor class: RandomMatrix
class RandomMatrix:
    def __init__(self, m, n, k, random_state=None, missing_fraction=0.1, noise_level=0):
        """
        Initializes the RandomMatrix generator.

        Parameters:
        - m: Number of rows.
        - n: Number of columns.
        - k: Rank of the matrix.
        - random_state: Seed for reproducibility.
        - missing_fraction: Fraction of entries to remove (set to 0).
        - noise_level: Standard deviation of Gaussian noise (set to 0 for no noise).
        """
        super().__init__()
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.rank = k  # Rank of the matrix
        self.random_state = random_state  # Random state for reproducibility
        self.missing_fraction = missing_fraction  # Fraction of missing entries
        self.noise_level = noise_level  # Noise level for Gaussian noise

        self.Gr, self.Lr = self.build_laplacians()

    # def build_laplacians(self):
    #     """
    #     Build row and column Laplacian matrices with weighted adjacency matrices.

    #     Parameters:
    #     - sigma: Bandwidth parameter for Gaussian kernel
    #     - k: Number of nearest neighbors (optional for sparsity)

    #     Returns:
    #     - Lr: Row Laplacian matrix (m x m)
    #     - Lc: Column Laplacian matrix (n x n)
    #     """
    #     # Generate random feature vectors for nodes (synthetic example)
    #     # row_features = np.random.rand(self.m, self.rank)  # Features for rows
    #     # col_features = np.random.rand(self.n, self.rank)  # Features for columns

    #     def truncated_normal(mean, std_dev, lower, upper, size):
    #         a, b = (lower - mean) / std_dev, (
    #             upper - mean
    #         ) / std_dev  # Truncation bounds
    #         return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

    #     # Parameters for the truncated normal distribution
    #     mean = 50  # Example mean
    #     std_dev = 1  # Example standard deviation
    #     lower, upper = 0.0, 100.0  # Example limits (e.g., [0, 1])

    #     # Generate features with truncated normal distribution
    #     row_features = truncated_normal(
    #         mean, std_dev, lower, upper, size=(self.m, self.rank)
    #     )
    #     col_features = truncated_normal(
    #         mean, std_dev, lower, upper, size=(self.n, self.rank)
    #     )

    #     # # Compute pairwise distances
    #     row_distances = squareform(pdist(row_features, metric="euclidean"))
    #     col_distances = squareform(pdist(col_features, metric="euclidean"))

    #     # Convert to sparse matrices
    #     adj_matrix_r_sparse = csr_matrix(row_distances)
    #     adj_matrix_c_sparse = csr_matrix(col_distances)

    #     # Compute Laplacians
    #     Lr = laplacian(adj_matrix_r_sparse, normed=True)
    #     Lc = laplacian(adj_matrix_c_sparse, normed=True)
    #     return Lr, Lc

    def build_laplacians(self):
        m = self.m
        size = m
        n_communities = int(np.sqrt(m) / 2)
        print(f"n_communities: {n_communities}")

        community_size = size // n_communities
        sizes = [community_size] * n_communities
        if size % n_communities:
            sizes[-1] += size % n_communities
        # Probability parameters for sparse community structure
        p_in = 0.7
        p_out = (1 - p_in) / n_communities  # Very sparse between-community connections
        # Generate probability matrix
        p_matrix = [
            [p_in if i == j else p_out for j in range(n_communities)]
            for i in range(n_communities)
        ]
        # Create graph using stochastic block model
        G = nx.stochastic_block_model(sizes, p=p_matrix)
        # Get adjacency matrix and compute Laplacian
        A = nx.adjacency_matrix(G).toarray()
        D = np.diag(np.sum(A, axis=1))
        L = D - A

        return G, L

    def build_A_matrices_from_laplacians(self, p=2):
        """
        Build A_r and A_c matrices using Laplacian eigendecomposition
        following equation (52): A_r = U_r g(Λ_r), A_c = U_c g(Λ_c)
        """
        # Compute eigendecomposition of Laplacian matrices
        Lambda_r, U_r = np.linalg.eigh(self.Lr)
        # Lambda_c, U_c = np.linalg.eigh(self.Lc)

        # Sort eigenvalues and eigenvectors in ascending order
        # idx_r = np.argsort(Lambda_r)
        # idx_c = np.argsort(Lambda_c)
        # Lambda_r = Lambda_r[idx_r]
        # Lambda_c = Lambda_c[idx_c]
        # U_r = U_r[:, idx_r]
        # U_c = U_c[:, idx_c]

        # Function g acting element-wise on eigenvalues
        # Apply g to all eigenvalues while avoiding small values
        # g_transformed_r = np.where(
        #     np.abs(Lambda_r) > 1e-10, np.power(Lambda_r, -p), Lambda_r
        # )
        # g_transformed_c = np.where(
        #     np.abs(Lambda_c) > 1e-10, np.power(Lambda_c, -p), Lambda_c
        # )
        g_transformed_r = np.where(
            np.abs(Lambda_r) > 1e-10, np.power(Lambda_r, -p), Lambda_r
        )
        # g_transformed_r = Lambda_r
        # g_transformed_c = Lambda_c

        # Compute A matrices according to equation (52)
        A_r = U_r @ np.diag(g_transformed_r)
        # A_c = U_c @ np.diag(g_transformed_c)
        # print(f"U {U_r}")
        # print(f"Lambda g {np.max(g_transformed_c)}")
        # print(f" A {A_r}")

        return A_r

    def generate_with_graphs(self, A_r):
        F = np.random.randn(self.m, self.rank)
        Q = np.random.randn(self.n, self.rank)

        Z = F @ Q.T
        # Z = np.eye(self.m, self.n)
        X = A_r @ Z

        total_entries = self.m * self.n
        missing_entries = int(total_entries * self.missing_fraction)

        missing_indices = np.random.choice(
            total_entries, missing_entries, replace=False
        )
        M_missing = X.copy()
        M_missing.flat[missing_indices] = 0
        mask = M_missing != 0  # Mask indicating non-missing entries
        return self, X, mask, self.Lr, np.zeros(self.n)

    def check_sim(self, X):
        row_sim = squareform(pdist(X, metric="euclidean"))
        print("row sim")
        print(row_sim)
        adj_Gr = nx.adjacency_matrix(self.Gr).toarray()
        print("similarties build on the adj")
        print(squareform(pdist(adj_Gr, metric="euclidean")))
        print("---------------------")

        col_sim = squareform(pdist(X.T, metric="euclidean"))
        print("col sim")
        print(col_sim)
        adj_Gc = nx.adjacency_matrix(self.Gc).toarray()
        print("similarties build on the adj")
        print(squareform(pdist(adj_Gc, metric="euclidean")))
        print("---------------------")

    def generate(self):
        """
        Generates a random low-rank matrix with missing values and optional Gaussian noise.

        Returns:
        - M_true: Low-rank matrix (m x n).
        - M_missing: Matrix with missing values (m x n).
        - M_noisy (optional): Matrix with Gaussian noise added to non-missing entries (m x n).
                             Returned only if noise_level > 0.
        """
        # Step 1: Create a low-rank matrix
        if self.random_state is not None:
            np.random.seed(self.random_state)

        U = np.random.randn(self.m, self.rank)
        V = np.random.randn(self.n, self.rank)
        M_true = U @ V.T

        # Step 2: Remove entries randomly to create missing values
        M_missing = M_true.copy()

        total_entries = self.m * self.n
        missing_entries = int(total_entries * self.missing_fraction)

        missing_indices = np.random.choice(
            total_entries, missing_entries, replace=False
        )
        M_missing.flat[missing_indices] = 0
        mask = M_missing != 0  # Mask indicating non-missing entries

        # Step 3: Add Gaussian noise if specified
        if self.noise_level > 0:
            noise = self.noise_level * np.random.randn(self.m, self.n)
            M_noisy = M_missing.copy()
            M_noisy[mask] += noise[mask]
            return M_true, mask, M_noisy

        return M_true, mask


# Successor class: DatasetMatrix
class DatasetMatrix:
    def __init__(
        self,
        dataset_name="ml-1m",
        rank=10,
        random_state=None,
    ):
        # Assuming dataset provides dimensions m and n
        self.m, self.n = self._get_dataset_dimensions(dataset_name)
        self.rank = rank
        self.dataset_name = dataset_name

    def _get_dataset_dimensions(self, dataset_name):
        """
        Retrieves the dimensions of the dataset.

        Placeholder logic: Replace with actual dataset dimension retrieval.
        For example: if dataset_name == 'ml-1m', return (6040, 3706).
        """
        if dataset_name == "ml-1m":
            return (6040, 3706)  # Example dimensions for MovieLens-1M dataset
        else:
            raise ValueError("Unknown dataset name")

    def parse_dataset(self):
        """
        Parses the dataset to populate M_true with its data.

        Placeholder logic: Replace with actual parsing logic.
        """
        pass

    def generate(self):
        """
        Generates a noisy low-rank matrix based on the dataset.
        """
        self.parse_dataset()


class SpotifyDataset:

    def __init__(
        self, src_path, min_playlist_len, min_num_tracks, preprocessed_path=None
    ):

        self.min_playlist_len = min_playlist_len
        self.min_num_tracks = min_num_tracks

        if preprocessed_path is not None:
            self.data = np.load(preprocessed_path)
        else:
            self.data = self.__construct_ds__(src_path)

    def get_data(self):
        return self.data

    def __construct_ds__(self, path):

        playlist_to_tracks = defaultdict(list)
        track_to_playlists = defaultdict(list)

        for f_name in tqdm(os.listdir(path)):
            with open(join(path, f_name), "r") as f:
                data = json.load(f)

                for playlist in data["playlists"]:
                    playlist_id = playlist["pid"]
                    for track in playlist["tracks"]:
                        track_id = hash(track["track_uri"])

                        playlist_to_tracks[playlist_id].append(track_id)
                        track_to_playlists[track_id].append(playlist_id)

        deleting_tracks = []

        for track, playlists in tqdm(
            track_to_playlists.items(), total=len(track_to_playlists)
        ):
            if len(playlists) < self.min_playlist_len:  # 3000
                deleting_tracks.append(track)

        for track in deleting_tracks:
            del track_to_playlists[track]

        new_playlist_to_tracks = {}
        for playlist, tracks in tqdm(
            playlist_to_tracks.items(), total=len(playlist_to_tracks)
        ):
            new_tracks = [track for track in tracks if track in track_to_playlists]
            if len(new_tracks) >= self.min_num_tracks:  # 100
                new_playlist_to_tracks[playlist] = new_tracks

        playlist_to_tracks = new_playlist_to_tracks

        playlist_to_id = {playlist: i for i, playlist in enumerate(playlist_to_tracks)}
        track_to_id = {track: i for i, track in enumerate(track_to_playlists)}

        playlist_to_tracks_id = [None] * len(playlist_to_tracks)
        track_to_playlists_id = [None] * len(track_to_playlists)

        for playlist, tracks in playlist_to_tracks.items():
            playlist_id = playlist_to_id[playlist]
            track_ids = [track_to_id[track] for track in tracks]
            playlist_to_tracks_id[playlist_id] = track_ids

        for track, playlists in track_to_playlists.items():
            track_id = track_to_id[track]
            playlist_ids = [
                playlist_to_id[playlist]
                for playlist in playlists
                if playlist in playlist_to_id
            ]
            track_to_playlists_id[track_id] = playlist_ids

        self.data = np.zeros((len(playlist_to_tracks), len(track_to_playlists)))

        for i, item in enumerate(playlist_to_tracks_id):
            for j in item:
                self.data[i][j] = 1
