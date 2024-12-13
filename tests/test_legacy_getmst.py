import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mistreeplus.legacy import GetMST  # Replace 'yourmodule' with the actual module name containing GetMST

@pytest.fixture
def mock_coords():
    """Mock the coords module."""
    with patch("mistreeplus.coords") as coords:
        coords.usphere2cart.return_value = (np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))
        coords.sphere2cart.return_value = (np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))
        coords.usphere_dist2ang.return_value = np.array([0.1, 0.2, 0.3])
        yield coords

@pytest.fixture
def mock_graph():
    """Mock the graph module."""
    with patch("mistreeplus.graph") as graph:
        graph.construct_knn2D.return_value = "mock_knn2D_graph"
        graph.construct_knn3D.return_value = "mock_knn3D_graph"
        graph.graph2data.return_value = (np.array([[0, 1], [1, 2]]), np.array([0.1, 0.2]))
        yield graph

@pytest.fixture
def mock_mst():
    """Mock the mst module."""
    with patch("mistreeplus.mst") as mst:
        mst.construct_mst.return_value = "mock_mst_graph"
        mst.get_edge_index.return_value = np.array([[0, 1], [1, 2]])
        mst.get_graph_degree.return_value = np.array([1, 2, 1])
        mst.get_stat_index.return_value = np.array([[1, 2], [2, 3]])
        yield mst

@pytest.fixture
def mock_branches():
    """Mock the branches module."""
    with patch("mistreeplus.legacy.branches") as branches:
        branches.find_branches.return_value = (["branch1", "branch2"], "rejected_branches")
        branches.get_branch_weight.return_value = np.array([0.5, 0.8])
        branches.get_branch_shape.return_value = np.array([0.7, 0.9])
        yield branches

def test_initialization_2D():
    """Test initialization with 2D Cartesian coordinates."""
    mst = GetMST(x=np.array([0, 1]), y=np.array([1, 2]))
    assert mst._mode == '2D'
    assert np.array_equal(mst.x, np.array([0, 1]))
    assert np.array_equal(mst.y, np.array([1, 2]))

def test_initialization_3D():
    """Test initialization with 3D Cartesian coordinates."""
    mst = GetMST(x=np.array([0, 1]), y=np.array([1, 2]), z=np.array([2, 3]))
    assert mst._mode == '3D'

def test_construct_mst_2D(mock_coords, mock_graph, mock_mst):
    """Test MST construction for 2D mode."""
    x = np.arange(50)
    y = np.arange(50)+1
    mst = GetMST(x=x, y=y)
    mst.construct_mst()
    assert mst.edge_length is not None

def test_construct_mst_3D(mock_coords, mock_graph, mock_mst):
    """Test MST construction for 3D mode."""
    x = np.arange(50)
    y = np.arange(50)+1
    z = np.arange(50)+2
    mst = GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    assert mst.edge_length is not None

def test_get_degree(mock_coords, mock_graph, mock_mst):
    """Test calculation of node degrees."""
    mst = GetMST(x=np.array([0, 1, 2]), y=np.array([1, 2, 3]))
    mst.k_neighbours = 1
    mst.construct_mst()
    mst.get_degree()
    assert np.array_equal(mst.degree, np.array([1, 2, 1]))

def test_get_degree_raises_error():
    """Test that get_degree raises an error if edge_index is undefined."""
    mst = GetMST(x=np.array([0, 1]), y=np.array([1, 2]))
    with pytest.raises(ValueError, match="minimum spanning tree has yet to be constructed"):
        mst.get_degree()

def test_clean():
    """Test the clean method."""
    mst = GetMST(x=np.array([0, 1]), y=np.array([1, 2]))
    mst.clean()
    assert mst.x is None
    assert mst._mode is None

def test_get_branches(mock_coords, mock_graph, mock_mst, mock_branches):
    """Test branch finding functionality."""
    x = np.arange(50)
    y = np.arange(50)+1
    mst = GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_branches()
    assert mst.branch_index != None

def test_get_stats(mock_coords, mock_graph, mock_mst, mock_branches):
    """Test the full pipeline of getting MST stats."""
    x = np.arange(50)
    y = np.arange(50)+1
    mst = GetMST(x=x, y=y)
    stats = mst.get_stats()
    assert len(stats) == 4  # degree, edge_length, branch_length, branch_shape
