import numpy as np
import pytest

from synthfin.correlation import (
    NaiveCorrelationGenerator,
    HierarchicalCorrelationGenerator
)
from synthfin.utils import validate_correlation_matrix


class TestNaiveCorrelationGenerator:
    """
    Test suite for NaiveCorrelationGenerator.
    """
    
    def test_initialization(self):
        """
        Test generator initialization.
        """
        gen = NaiveCorrelationGenerator(n=10)
        assert gen.n == 10
    
    def test_generate_shape(self):
        """
        Test generated matrix has correct shape.
        """
        n = 15
        gen = NaiveCorrelationGenerator(n=n)
        matrix = gen.generate()
        assert matrix.shape == (n, n)
    
    def test_correlation_properties(self):
        """
        Test generated matrix has valid correlation properties.
        """
        gen = NaiveCorrelationGenerator(n=20)
        matrix = gen.generate()
        
        # Check diagonal is 1
        assert np.allclose(np.diag(matrix), 1.0)
        
        # Check symmetric
        assert np.allclose(matrix, matrix.T)
        
        # Check values in [-1, 1]
        assert np.all(matrix >= -1.0 - 1e-10)  # Small tolerance for numerical errors
        assert np.all(matrix <= 1.0 + 1e-10)
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        # More tolerant check for positive semi-definiteness
        # Small negative eigenvalues (up to -1e-8) are acceptable due to numerical precision
        assert min_eigenvalue >= -1e-8, f"Minimum eigenvalue {min_eigenvalue} is too negative"
        
        # If there are small negative eigenvalues, they should be very small
        if min_eigenvalue < 0:
            assert abs(min_eigenvalue) < 1e-6, f"Negative eigenvalue {min_eigenvalue} is too large"
    
    def test_validate_correlation_matrix(self):
        """
        Test correlation matrix validation.
        """
        gen = NaiveCorrelationGenerator(n=10)
        matrix = gen.generate()
        assert validate_correlation_matrix(matrix)


class TestHierarchicalCorrelationGenerator:
    """
    Test suite for HierarchicalCorrelationGenerator.
    """
    
    def test_initialization_default(self):
        """
        Test generator initialization with defaults.
        """
        n = 20
        gen = HierarchicalCorrelationGenerator(n=n)
        assert gen.n == n
        assert gen.n_clusters == int(np.sqrt(n))
    
    def test_initialization_custom(self):
        """
        Test generator initialization with custom parameters.
        """
        gen = HierarchicalCorrelationGenerator(
            n=30,
            n_clusters=5,
            intra_cluster_corr=0.8,
            inter_cluster_corr=0.1,
            noise_level=0.05
        )
        assert gen.n == 30
        assert gen.n_clusters == 5
        assert gen.intra_cluster_corr == 0.8
    
    def test_hierarchical_structure(self):
        """
        Test that hierarchical structure is created correctly.
        """
        n = 20
        n_clusters = 4
        gen = HierarchicalCorrelationGenerator(
            n=n,
            n_clusters=n_clusters,
            intra_cluster_corr=0.9,
            inter_cluster_corr=0.1,
            noise_level=0.01  # Low noise to test structure
        )
        matrix = gen.generate()
        
        # Check that intra-cluster correlations are higher
        cluster_size = n // n_clusters
        
        # Get average intra-cluster correlation (excluding diagonal)
        intra_corrs = []
        for i in range(n_clusters):
            start = i * cluster_size
            end = min(start + cluster_size, n)
            cluster_matrix = matrix[start:end, start:end]
            # Get upper triangle excluding diagonal
            upper_tri = np.triu_indices_from(cluster_matrix, k=1)
            intra_corrs.extend(cluster_matrix[upper_tri])
        
        avg_intra = np.mean(intra_corrs)
        
        # Get average inter-cluster correlation
        inter_corrs = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                start_i = i * cluster_size
                end_i = min(start_i + cluster_size, n)
                start_j = j * cluster_size
                end_j = min(start_j + cluster_size, n)
                
                block = matrix[start_i:end_i, start_j:end_j]
                inter_corrs.extend(block.flatten())
        
        avg_inter = np.mean(inter_corrs)
        
        # Intra-cluster should be significantly higher than inter-cluster
        assert avg_intra > avg_inter + 0.3
    
    def test_correlation_properties(self):
        """
        Test generated matrix has valid correlation properties.
        """
        gen = HierarchicalCorrelationGenerator(n=25, n_clusters=5)
        matrix = gen.generate()
        
        assert validate_correlation_matrix(matrix)
    
    @pytest.mark.parametrize("n,n_clusters", [
        (10, 2),
        (15, 3),
        (20, 4),
        (25, 5),
        (30, 6),
    ])
    def test_various_sizes(self, n, n_clusters):
        """
        Test generation with various sizes.
        """
        gen = HierarchicalCorrelationGenerator(n=n, n_clusters=n_clusters)
        matrix = gen.generate()
        assert matrix.shape == (n, n)
        assert validate_correlation_matrix(matrix)