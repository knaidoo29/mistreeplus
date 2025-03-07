# MiSTree+

|               |                                           |
|---------------|-------------------------------------------|
| Author        | Krishna Naidoo                            |          
| Version       | 0.1.0                                     |
| Repository    | https://github.com/knaidoo29/mistreeplus  |
| Documentation | TBA                                       |

## Why the +?

``MiSTree+`` is a complete rewrite of the ``MiSTree`` ``python`` package, designed with the intent of being more flexible and powerful to its predecessor ``MiSTree``. It includes methods for computing the MST with a Delaunay tesselation as well as providing new statistical methods for analysing the resulting tree statistically or for topology. We have opted for maintaining ``MiSTree``, hence the ``+``, so that we can make broad changes without breaking exiting pipelines built around the original ``MiSTree``.

## Introduction

The *minimum spanning tree* (MST), a graph constructed from a distribution of points, draws lines between pairs of points so that all points are linked in a single skeletal structure that contains no loops and has minimal total edge length. The MST has been used in a broad range of scientific fields such as particle physics, in astronomy and cosmology. Its success in these fields has been driven by its sensitivity to the spatial distribution of points and the patterns within.

``MiSTree+``, a public ``Python`` package, allows a user to construct the MST in a variety of coordinates systems, including celestial coordinates (RA and Dec) used in astronomy. The package enables the MST to be constructed quickly by initially using a *k*-nearest neighbour graph (*k* NN, rather than a matrix of pairwise distances) or Delaunay tesselation, which is then fed to Kruskal's algorithm to construct the MST. 

``MiSTree+`` enables a user to measure the statistics of the MST and provides classes for binning the MST statistics (into histograms) and plotting the distributions. Applying the MST will enable the inclusion of high-order statistics information from the cosmic web which can provide additional information to improve cosmological parameter constraints. This information has not been fully exploited due to the computational cost of calculating *N*-point statistics. ``MiSTree+`` was designed to be used in cosmology but could be used in any field which requires extracting non-Gaussian information from point distributions.

## Dependencies

* Python >= 3.7
* `numpy`
* `numba`
* `matplotlib`
* `scipy`
* `scikit-learn`

For testing you will require `nose` or `pytest`.


## Installation

MiSTree can be installed as follows:

```
pip install mistreeplus [--user]
```
The `--user` is optional and only required if you don’t have write permission. If you
are using a windows machine this may not work, in this case (or as an alternative to pip) clone the repository,

```
git clone https://github.com/knaidoo29/mistreeplus.git
cd mistreeplus
```

and install by either running

```
pip install . [--user]
```

or

```
python setup.py build
python setup.py install
```

Similarly, if you would like to work and edit `mistreeplus` you can clone the repository and install an editable version:

```
git clone https://github.com/knaidoo29/mistreeplus.git
cd mistreeplus
pip install -e . [--user]
```

From the `mistreeplus` directory you can then test the install using `nose`:

```
python setup.py test
```

or using `pytest`:

```
python -m pytest
```

You should now be able to import the module:

```python
import mistreeplus as mist
```

## Notes

### Spherical Coordinate Conventions

Spherical coordinates are defined in MiSTree according to two conventions which will be referred
to as the spherical polar coordinates and the celestial coordinates.

1. Spherical polar coordinates: The angular components are given by the parameters `phi`
and `theta`.
  - `phi` is the longitude parameter belonging in the range [0, 360] degrees.
  - `theta` is the latitude parameter belonging in the range [0, 180] degrees where `theta=0` lies at the north pole.
2. Celestial coordinates: The angular components are given by the astronomy parameters Right Ascension (`ra`)
and Declination (`dec`).
  - `ra` is the longitude parameter belonging in the range [0, 360] degrees.
  - `dec` is the latitude parameter belonging in the range [-90, 90] degrees.

To switch between the two conventions simply requires the following (given in degrees):

```
  ra = phi
  dec = 90. - theta
```

All angular coordinates can be provided either in degrees or radians which can be specified
by setting `units` to `degs` for degrees and `rads` for radians.

### Minimum Spanning Tree

#### Ensuring the constructed MST is spanning

In version 1 of MiSTree the minimum spanning tree (MST) was determined from a
k-nearest neighbour graph (kNN). We used this as our input to increase the speed
of the MST construction but this choice can lead to a spanning tree that is not
spanning. In these cases a spanning tree can usually be determined by increasing
the value of `k` to include many more neighbours. This is not an ideal solution
as in certain cases this will still not resolve the issue. In version 2 of MiSTree
we now by default construct this input tree using the Delaunay triangulation. Since
the edges of the MST are by definition members of the Delaunay triangulation it
ensures that the constructed MST is the true MST and not an approximation as was
previously the case. Users of version 1 will still be able to use kNN if they wish
or the can input a tree of their own which perhaps links other properties.

## Functions

This is an exhaustive list of all functions and classes provided in MiSTree.


* `check` : Performs sanity checks to ensure things are as expected.
  - `check_angle_units` : Checks angle units is either `degs` or `rads`.
  - `check_phi_in_range` : Checks `phi` is within range.
  - `check_theta_in_range` : Checks `theta` is within range.
  - `check_ra_in_range` : Check `ra` is within range.
  - `check_dec_in_range` : Check `dec` is within range.
  - `check_r_unit_sphere` : Check `r` is consistent with a unit sphere.
  - `check_length` : Checks the length of an array.
  - `check_positive` : Checks values are positive.
  - `check_finite` : Checks where values are finite.
  - `check_levy_mode` : Checks spatial mode for Levy flight.

* `coords` : Houses a bunch of functions designed to deal with different coordinate
systems and the transformations from one coordinate system to another.
  - `dist2D` : Calculates distance between points in 2D.
  - `dist3D` : Calculates distance between points in 3D.
  - `dec2theta` : Converts celestial `dec` to spherical polar `theta`.
  - `theta2dec` : Converts spherical polar `theta` to celestial `dec`.
  - `sphere2cart` : Converts spherical polar coordinates to cartesian coordinates.
  - `cart2sphere` : Converts cartesian coordinates to spherical polar coordinates.
  - `sphere2cart_radec` : Converts celestial coordinates to cartesian coordinates.
  - `cart2sphere_radec` : Converts cartesian coordinates to celestial coordinates.
  - `usphere2cart` : Converts spherical polar coordinates on a unit sphere to cartesian coordinates.
  - `cart2usphere` : Converts cartesian coordinates to spherical polar coordinates on a unit sphere.
  - `usphere2cart_radec` : Converts spherical polar coordinates on a unit sphere to cartesian coordinates.
  - `cart2usphere_radec` : Converts cartesian coordinates to spherical polar coordinates on a unit sphere.
  - `xy2vert` : Stacks x and y coordinates to a vertices format.
  - `vert2xy` : Unstacks vertices to x and y coordinates.
  - `xyz2vert` : Stacks x, y and z coordinates to a vertices format.
  - `vert2xyz` : Unstacks vertices to x, y and z coordinates.

* `graph` : Graph based functions.
  - `graph2data` : Returns the node index and weights of a graph given in `csr_matrix` (scipy sparse matrix) format.
  - `data2graph` : Returns a graph in `csr_matrix` format given edge node indices and edge weights.
  - `construct_delaunay2D` : Constructs Delaunay triangulation graph in 2D.
  - `construct_delaunay3D` : Constructs Delaunay triangulation graph in 3D.
  - `construct_knn2D` : Constructs k-Nearest Neighbour graph in 2D.
  - `construct_knn3D` : Constructs k-Nearest Neighbour graph in 3D.

* `legacy`: Legacy  functions for computing the degree, edge length, branch length and shape statistics computed by `mistree`.
  - `find_branches`: Finds branches in MST.
  - `get_branch_weight`: Finds branch weights.
  - `get_branch_end_index`: Finds the node index of branch ends.
  - `get_branch_edge_count`: Count the number of edges in each branch.
  - `get_branch_shape`: Finds the shape of branches.
  - `GetMST`: A lightweight replacement to the `mistree.GetMST` class, useful for comparison or for reproducing `mistree` outputs.
  - `get_edge_index`: Combined edge index arrays.
  - `get_stat_index`: Create a 2D array, with the stat property of each node at the end of each edge.
  - `get_degree`: Gets the degree for each node.

* `levy` : Levy flight random walk samples.
  - `generate_user_flight`: Generates random walk samples with user defined steps.
  - `generate_levy_flight`: Generates Levy flight samples.
  - `generate_adj_levy_flight`: Generates the adjusted Levy flight sample.
  - `generate_levy_steps`: Generates steps following Levy flight step distribution.
  - `generate_adj_levy_steps`: Generates steps following adjusted Levy flight step distributions.

* `mst` : Minimum Spanning Tree functions.
  - `construct_mst` : Constructs the MST of an input graph.

* `randoms` : Generates randoms.
  - `cart1d` : Generates a uniform set of randoms in 1D.
  - `cart2d` : Generates a uniform set of randoms in 2D.
  - `cart3d` : Generates a uniform set of randoms in 3D.
  - `polar_r` : Generates random radial in polar coordinates.
  - `polar_phi` : Generates random phi in polar coordinates.
  - `polar_rphi` : Generates random in polar coordinates.
  - `usphere_phi` : Generates random phis on a unit sphere.
  - `usphere_theta` : Generates random thetas on a unit sphere.
  - `usphere_phitheta` : Generates randoms on a unit sphere.
  - `sphere_r` : Generates random radial in spherical polar coordinates.
  - `sphere_phi` : Generates random phi values in spherical polar coordinates.
  - `sphere_theta` : Generates random theta values in spherical polar coordinates.
  - `sphere_rphitheta` : Generates randoms in spherical polar coordinates.

* `src` : Numba fast JIT compiled code, formally written in Fortran.
  - `dotvector3` : Dot product of two vectors of length 3.
  - `dot3by3mat3vec` : Dot product of 3 by 3 matrix with a vector of length 3.
  - `crossvector3` : Cross product of two vectors of length 3.
  - `normalisevector` : Normalise a vector.
  - `inv3by3` : Inverts 3 by 3 matrix.
  - `getgraphdegree` : Returns the node degrees for an input graph.
  - `periodicboundary` : Ensures points are within a periodic box.
  - `randwalkcart2d` : Random walk simulation in 2D.
  - `randwalkcart3d` : Random walk simulation in 3D.
  - `usphererotate` : Rotates point on unit sphere.
  - `randwalkusphere` : Random walk simulation on a unit sphere.

* `tree` : Functions for finding adjacent nodes, constructing tree dictionaries and finding paths in a tree.
  - `get_adjacents` : Find the adjacent points to each node.
  - `smooth_stat_with_graph` : Smoothes a statistics based on adjacent indices.
  - `get_edge_dict` : Constructs an edge node dictionary to call weight values quickly.
  - `get_groups` : Finds groups in a disconnected graph.
  - `perc_from_root_by_N` : Finds percolation paths of length N from a defined root node.
  - `perc_from_all_by_N` : Finds percolation paths of length N from all nodes.
  - `percpath2weight` : Finds the weight of percolation paths.
  - `percpath2percends` : Finds the ends of each percolation path.
  - `percend_dist2D` : Finds the distance of percolation path ends in 2D.
  - `percend_dist3D` : Finds the distance of percolation path ends in 2D.
  - `adjacents2tree` : Converts adjacents list to a tree structured dictionary.
  - `findpath2root` : Finds the path for a node to the root node of a tree.
  - `findpath` : Finds the path across a tree between any points on a node.
  - `get_path_weight` : Finds the weight of a path.
  - `get_centrality` : Defines the centrality of a graph.
  - `get_spine` : Finds the spine from a specific starting node, determined by its centrality.
  - `get_spines` : Finds the spines of a given tree.

## Support

* Bugs and issues should be reported via github issues [here](https://github.com/knaidoo29/mistreeplus/issues).
* Suggestion for more functions or new statistics can be emailed to _krishna.naidoo.11@ucl.ac.uk_.

## Versions

* **Version 0.1.0 [07/03/25]**: Pre-release of the MiSTree+ python package, with the addition of percolation statistics.

* **Version 0.0.0 [09/12/24]**: First pre-release of MiSTree+, which retains all the functionalities of MiSTree but with a different framework in place. Tasks have been separated so that different aspects of the code are easier to manage and understand.