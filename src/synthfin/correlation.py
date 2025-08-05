from abc import ABC, abstractmethod
import warnings

import numpy as np
from scipy.stats import random_correlation


class CorrelationMatrixGenerator(ABC):
    """
    Abstract base class for correlation matrix generators.
    """
    
    def __init__(self, n: int, **kwargs):
        """
        Initialize the correlation matrix generator.
        
        Args:
            n: Size of the correlation matrix (n x n)
            **kwargs: Additional keyword arguments for specific implementations
        """
        self.n = n
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate an n x n correlation matrix.
        """
        pass
    
    def _make_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure matrix is positive semidefinite and valid correlation matrix.
        """
        # Method 1: Eigenvalue decomposition with proper scaling
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Set negative eigenvalues to small positive value
        eigenvalues[eigenvalues < 0] = 1e-8
        
        # Reconstruct matrix
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to ensure it's a correlation matrix
        # Extract diagonal elements
        d = np.sqrt(np.diag(matrix))
        
        # Avoid division by zero
        d[d == 0] = 1
        
        # Normalize
        matrix = matrix / np.outer(d, d)
        
        # Ensure diagonal is exactly 1 and matrix is symmetric
        np.fill_diagonal(matrix, 1.0)
        matrix = (matrix + matrix.T) / 2
        
        return matrix


class NaiveCorrelationGenerator(CorrelationMatrixGenerator):
    """
    Naive correlation matrix generator that creates random valid correlation matrices.
    Uses a method that guarantees positive semi-definiteness.
    """
    
    def __init__(self, n: int, random_factor: int = None, **kwargs):
        """
        Initialize the naive correlation generator.
        
        Args:
            n: Size of the correlation matrix
            random_factor: Factor for random matrix generation (default: n + 50)
        """
        super().__init__(n, **kwargs)
        self.random_factor = random_factor or max(n + 50, 2 * n)
    
    def generate(self) -> np.ndarray:
        """
        Generate a random valid correlation matrix.
        """
        # Method: Generate random factor loadings and create correlation from them
        # This guarantees a valid correlation matrix
        
        # Generate random factor loadings matrix
        # More rows than columns ensures positive definiteness
        W = np.random.randn(self.n, self.random_factor)
        
        # Create covariance matrix
        S = W @ W.T
        
        # Add small diagonal term for numerical stability
        S += np.eye(self.n) * 1e-6
        
        # Convert to correlation matrix
        # Extract standard deviations
        std_devs = np.sqrt(np.diag(S))
        
        # Normalize to get correlation matrix
        corr_matrix = S / np.outer(std_devs, std_devs)
        
        # Ensure exact properties
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Ensure perfect symmetry
        
        # Clip any numerical errors
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        return corr_matrix


class HierarchicalCorrelationGenerator(CorrelationMatrixGenerator):
    """
    Generates correlation matrices with hierarchical clustering structure.
    This creates blocks of higher correlations to simulate sector/industry clustering.
    """
    
    def __init__(self, n: int, n_clusters: int = None, intra_cluster_corr: float = 0.7,
                 inter_cluster_corr: float = 0.2, noise_level: float = 0.1, **kwargs):
        """
        Initialize the hierarchical correlation generator.
        
        Args:
            n: Size of the correlation matrix
            n_clusters: Number of clusters (default: sqrt(n))
            intra_cluster_corr: Base correlation within clusters
            inter_cluster_corr: Base correlation between clusters
            noise_level: Amount of random noise to add
        """
        super().__init__(n, **kwargs)
        self.n_clusters = n_clusters or int(np.sqrt(n))
        self.intra_cluster_corr = intra_cluster_corr
        self.inter_cluster_corr = inter_cluster_corr
        self.noise_level = noise_level
    
    def generate(self) -> np.ndarray:
        """
        Generate a hierarchical correlation matrix.
        """
        # Initialize with inter-cluster correlation
        matrix = np.full((self.n, self.n), self.inter_cluster_corr)
        
        # Assign assets to clusters
        cluster_sizes = [self.n // self.n_clusters] * self.n_clusters
        # Distribute remaining assets
        for i in range(self.n % self.n_clusters):
            cluster_sizes[i] += 1
        
        # Create intra-cluster correlations
        start_idx = 0
        for cluster_size in cluster_sizes:
            end_idx = start_idx + cluster_size
            matrix[start_idx:end_idx, start_idx:end_idx] = self.intra_cluster_corr
            start_idx = end_idx
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, size=(self.n, self.n))
        noise = (noise + noise.T) / 2  # Make symmetric
        matrix += noise
        
        # Ensure correlations are in [-1, 1]
        matrix = np.clip(matrix, -1, 1)
        
        # Set diagonal to 1
        np.fill_diagonal(matrix, 1.0)
        
        # Make positive semidefinite
        matrix = self._make_positive_semidefinite(matrix)
        
        return matrix