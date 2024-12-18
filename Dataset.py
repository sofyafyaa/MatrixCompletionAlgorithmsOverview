import numpy as np


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
            return rm.generate()


# Successor class: RandomMatrix
class RandomMatrix:
    def __init__(
        self, m, n, k, random_state=None, missing_fraction=0.1, noise_level=0.0
    ):
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
            return M_true, missing_indices, M_noisy

        return M_true, missing_indices


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
