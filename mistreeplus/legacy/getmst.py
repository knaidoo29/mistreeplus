import numpy as np
from typing import Optional

from . import branches
from . import stats

from .. import mst
from .. import coords
from .. import graph


class GetMST:
    """
    A lightweight GetMST function to reproduce the statistics from the mistree
    GetMST class.
    """

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        ra: Optional[np.ndarray] = None,
        dec: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        units : str = 'deg'
    ):
        """
        Parameters
        ----------
        x, y, (z) : array
            Cartesian 2D (3D) coordinates.
        phi, theta, (r) : array
            Tomographic (spherical) coordinates.
        ra, dec, (r) : array
            Celestial tomographic (spherical) coordinates.
        units : {'deg', 'rad'}, optional
            The units of the celestial coordinates ra and dec.

        Notes
        -----
        The default of all of the input parameters are set to 'None' such that an internal parameter '_mode'
        of the MST can be set based on the input parameters. Supply:
            * x and y - for 2D cartesian coordinate. "_mode='2D'"
            * x, y and z - for 3D cartesian coordinates.  "_mode='3D'"
            * phi and theta - for tomographic coordinates. "_mode='tomographic'"
            * phi, theta and r - for spherical polar coordinates. "_mode='spherical polar'"
            * ra and dec - for celestial coordinates. "_mode='tomographic celestial'"
            * ra, dec and r - for celestial spherical polar coordinates. "_mode='spherical polar celestial'"
        """
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.theta = theta
        self.ra = ra
        self.dec = dec
        self.r = r
        self.units = units
        if self.x is not None and self.y is not None:
            if self.z is None:
                self._mode = '2D'
            else:
                self._mode = '3D'
        elif self.phi is not None and self.theta is not None:
            if self.r is None:
                self._mode = 'usphere'
                self.x, self.y, self.z = coords.usphere2cart(
                    self.phi, self.theta, units=self.units
                )
            else:
                self._mode = 'sphere'
                self.x, self.y, self.z = coords.sphere2cart(
                    self.r, self.phi, self.theta, units=self.units
                )
        elif self.ra is not None and self.dec is not None:
            if self.r is None:
                self._mode = 'usphere'
                self.x, self.y, self.z = coords.usphere2cart_radec(
                    self.ra, self.dec, units=self.units
                )
            else:
                self._mode = 'sphere'
                self.x, self.y, self.z = coords.sphere2cart_radec(
                    self.r, self.ra, self.dec, units=self.units
                )
        self.k_neighbours = 20
        self.edge_length = None
        self.edge_index = None
        self.degree = None
        self.edge_degree = None
        self.branch_index = None
        self.branch_length = None
        self.branch_edge_count = None
        self.branch_shape = None


    def define_k_neighbours(self, k_neighbours: int):
        """
        Sets the k_neighbours value. This is automatically set to 20 if this is not called.

        Parameters
        ----------
        k_neighbours : int
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
        """
        self.k_neighbours = k_neighbours


    def construct_mst(self):
        """Constructs the minimum spanning tree from the input data set."""
        if self._mode == '2D':
            knn_graph = graph.construct_knn2D(self.x, self.y, self.k_neighbours)
        else:
            knn_graph = graph.construct_knn3D(self.x, self.y, self.z, self.k_neighbours)
        mst_graph = mst.construct_mst(knn_graph)
        ind1, ind2, self.edge_length = graph.graph2data(mst_graph)
        if self._mode == 'usphere':
            self.edge_length = coords.usphere_dist2ang(self.edge_length)
        self.edge_index = stats.get_edge_index(ind1, ind2)


    def get_degree(self):
        """Finds the degree of each node in the constructed MST."""
        if self.edge_index is not None:
            self.degree = stats.get_degree(self.edge_index, len(self.x))
        else:
            raise ValueError("'edge_index' are undefined, meaning the minimum spanning tree has yet to be constructed.")


    def get_degree_for_edges(self):
        """Gets the degree of the nodes at each end of all edge."""
        if self.degree is not None:
            self.edge_degree = stats.get_stat_index(self.edge_index, self.degree)
        else:
            raise ValueError("The degrees are undefined, meaning they have yet to be calculated.")


    def get_branches(self, sub_divisions: Optional[int] = None):
        """
        Finds the branches of a MST.

        Parameters
        ----------
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis.
            Used for speeding up the branch finding algorithm when using many
            points (> 100000).
        """
        if self._mode == '2D':
            branch_index, rejected_branch_index = branches.find_branches(
                self.edge_index, self.degree, x=self.x, y=self.y, div=sub_divisions
            )
        else:
            branch_index, rejected_branch_index = branches.find_branches(
                self.edge_index, self.degree, x=self.x, y=self.y, z=self.z
            )
        self.branch_index = branch_index
        self.branch_length = branches.get_branch_weight(self.branch_index, self.edge_length)

    def get_branch_edge_count(self):
        """Finds the number of edges included in each branch."""
        branch_edge_count = [float(len(i)) for i in self.branch_index]
        self.branch_edge_count = np.array(branch_edge_count)

    def get_branch_shape(self):
        """Finds the shape of all branches. This is simply the straight line distance between the two ends divided by
        the branch length."""
        if self._mode == '2D':
            self.branch_shape = branches.get_branch_shape(
                edge_ind=self.edge_index, edge_deg=self.edge_degree,
                branch_ind=self.branch_index, branch_weight=self.branch_length,
                mode="2D", x=self.x, y=self.y
            )
        elif self._mode == '3D' or self._mode == 'sphere':
            self.branch_shape = branches.get_branch_shape(
                edge_ind=self.edge_index, edge_deg=self.edge_degree,
                branch_ind=self.branch_index, branch_weight=self.branch_length,
                mode="3D", x=self.x, y=self.y, z=self.z
            )
        elif self._mode == 'usphere':
            self.branch_shape = branches.get_branch_shape(
                edge_ind=self.edge_index, edge_deg=self.edge_degree,
                branch_ind=self.branch_index, branch_weight=self.branch_length,
                mode="usphere", x=self.x, y=self.y, z=self.z
            )
        else:
            pass

    def output_stats(self, include_index: bool = False):
        """Outputs the MST statistics.

        Parameters
        ----------
        include_index : bool, optional
            If true will output the indexes of the nodes for each edge and the indexes of edges in each branch.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : array, optional
            A 2 dimensional array, where the first nested array shows the indexes for the nodes
            on one end of the edge and the second shows the other node.
        branch_index : list, optional
            A list of branches, where each branch is given as a list of the indexes of the member edges.
        """
        if include_index == True:
            return self.degree, self.edge_length, self.branch_length, self.branch_shape, self.edge_index, \
                   self.branch_index
        else:
            return self.degree, self.edge_length, self.branch_length, self.branch_shape

    def _get_stats(
        self,
        include_index: bool = False,
        sub_divisions: Optional[int]=None,
        k_neighbours: Optional[int]=None,
    ):
        """Computes the MST and outputs the statistics.

        Parameters
        ----------
        include_index : bool, optional
            If True will output the indexes of the nodes for each edge and the indexes of edges in each branch.
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis. Used for speeding up the branch
            finding algorithm when using many points (> 100000).
        k_neighbours : int, optional
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : array, optional
            A 2 dimensional array, where the first nested array shows the indexes for the nodes
            on one end of the edge and the second shows the other node.
        branch_index : list, optional
            A list of branches, where each branch is given as a list of the indexes of the member edges.

        Notes
        -----
        This will calculate all the MST statistics by putting the data set through the following functions:
            1) k_neighbours (if k_neighbours is specified)
            2) construct_mst
            3) get_degree
            4) get_degree_for_edges
            5) get_branches
            6) get_branch_shape
            7) output_stats
        """
        if k_neighbours is not None:
            self.define_k_neighbours(k_neighbours)
        self.construct_mst()
        self.get_degree()
        self.get_degree_for_edges()
        self.get_branches(sub_divisions=sub_divisions)
        self.get_branch_shape()
        return self.output_stats(include_index=include_index)

    def get_stats(
        self,
        include_index: Optional[int] = False,
        sub_divisions: Optional[int] = None,
        k_neighbours: Optional[int] = None
    ):
        """Gets the minimum spanning tree statistics of a partitioned data set. Same inputs as 'get_stats'.

        Parameters
        ----------
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis.
        k_neighbours : int, optional
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : list, optional
            A list of 2 dimensional arrays for the nodes in each group.
        branch_index : list, optional
            A list of list of branches, where each branch is given as a list of the indexes of the member edges.
        """
        return self._get_stats(
            include_index=include_index, sub_divisions=sub_divisions,
            k_neighbours=k_neighbours
        )

    def clean(self):
        self.x = None
        self.y = None
        self.z = None
        self.ra = None
        self.dec = None
        self.r = None
        self.units = None
        self._mode = None
        self.k_neighbours = 20
        self.phi = None
        self.theta = None
        self.edge_length = None
        self.edge_index = None
        self.degree = None
        self.edge_degree = None
        self.branch_index = None
        self.branch_length = None
        self.branch_edge_count = None
        self.branch_shape = None
