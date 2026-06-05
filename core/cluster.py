"""
Copyright (C) 2020-2026 Nico Marioni <nmarioni@seas.upenn.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
""" 

# Cluster analysis to identify connected domains of free volume spheres, which are either percolated domains or solvent domains depending on the value of args.solvent_name

import argparse
from igraph import Graph
import MDAnalysis.lib.distances as distances
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional, List

from utils.constants import FLOAT_TYPE

def perform_clustering_analysis(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        last_frame: bool, 
        cell: np.ndarray, 
        radii_arr: np.ndarray, 
        sol: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Handles Moore/Neumann clustering and percolation to identify connected domains.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        last_frame (bool): True if the current frame is the last frame in frame_ids.
        cell (np.ndarray): Array of simulation cell dimensions.
        radii_arr (np.ndarray): Radius of the largest free volume sphere centered on each voxel.
        sol (np.ndarray): Coordinates of all solvent atoms.
        
    Returns:
        Tuple[np.ndarray, float, float]: Array of radii of the largest free volume sphere centered on each voxel, maximum diameter, and time to perform calculation.
    """
    
    # Read in voxel_data
    vox_x, vox_y, vox_z = voxel_data['vox_x'], voxel_data['vox_y'], voxel_data['vox_z']
    l_x, l_y, l_z = voxel_data['l_x'], voxel_data['l_y'], voxel_data['l_z']
    indexed_type, linear_type, signed_linear_type = voxel_data['indexed_type'], voxel_data['linear_type'], voxel_data['signed_linear_type']
    N_sol = len(sol)                                                                                                                            # Number of solvent atoms
    
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_Cluster = time.perf_counter()
        print('\n##### Performing Clustering Analysis - Percolated/Solvent-Domain #####')
    else: time_Cluster = 0.0

    # Create an interconnected graph lattice of the voxelized system, where voxels are associated to each other through their 6 3x3x3 cube-face-center neighbors
    if args.clustering == 'Neumann':
        # Only calculate out half of the neighbors per voxel to prevent double-counting
        Neighborhood = np.array([[1, 0, 0],                                                                                                     #Neighborhood = np.array([[1, 0, 0], [-1, 0, 0],
                                 [0, 1, 0],                                                                                                     #                         [0, 1, 0], [ 0,-1, 0],
                                 [0, 0, 1]], dtype=linear_type)                                                                                 #                         [0, 0, 1], [ 0, 0,-1]], dtype=linear_type)
    # Create an interconnected graph lattice of the voxelized system, where voxels are associated to each other through their 26 3x3x3 cube neighbors
    elif args.clustering == 'Moore':
        # Only calculate out half of the neighbors per voxel to prevent double-counting
        Neighborhood = np.array([[ 0,  0,  1],                                                                                                  #Neighborhood = np.array([[-1, -1, -1], [-1, -1,  0], [-1, -1,  1],
                                 [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1],                                                                      #                         [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
                                 [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],                                                                      #                         [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],
                                 [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],                                                                      #                         [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1],
                                 [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]], dtype=signed_linear_type)                                           #                         [ 0,  0, -1], [ 0,  0,  1], # No [ 0,  0,  0]
                                                                                                                                                #                         [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1],
                                                                                                                                                #                         [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],
                                                                                                                                                #                         [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],
                                                                                                                                                #                         [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]], dtype=linear_type)
    else:
        raise ValueError(f'Clustering variable incorrectly set: {args.clustering}')
    
    # Create graph for cluster analysis, where each voxel is indexed sequentially, not in spatial (x,y,z) indices
    G = Graph(l_x * l_y * l_z, directed=False)

    # Retrieve index of all free volume spheres of radius r >= probe_radius and linearize their indices for use in the cluster graph analysis
    radii_arr = radii_arr.ravel()
    Graph_radii = radii_arr >= args.probe_radius                                                                                                # Linearized free volume sphere radii

    # For efficiency, we limit the number of edges generated per loop
    count = 0
    while count < len(radii_arr):
        count_old = count; count += min(int(args.N_edge_gen), len(radii_arr)-count_old)

        Graph_idx = np.where(Graph_radii[count_old:count])[0].astype(linear_type)                                                               # Linearized indices of free volume spheres
        idx_x, idx_y, idx_z = np.unravel_index(count_old + Graph_idx, (l_x, l_y, l_z))                                                          # Spatial indices of free volume spheres
        idx_x = idx_x.astype(indexed_type); idx_y = idx_y.astype(indexed_type); idx_z = idx_z.astype(indexed_type)

        # For memory efficiency, loop through each neighbor one-by-one
        for neigh in Neighborhood:
            # Retrieve linearized index of each voxel 'paired' to each free volume sphere voxel in Graph_idx
            edge_Graph_idx = (  ((idx_x + neigh[0]) % l_x) * l_y * l_z
                              + ((idx_y + neigh[1]) % l_y) * l_z
                              + ((idx_z + neigh[2]) % l_z)            )
            
            if edge_Graph_idx.dtype != Graph_idx.dtype: edge_Graph_idx = edge_Graph_idx.astype(linear_type)
            # Add 'pairs' (edges) to graph G, where we only consider voxel edges between free volume spheres
            G.add_edges(np.stack((
                                       Graph_idx[Graph_radii[edge_Graph_idx]],
                                  edge_Graph_idx[Graph_radii[edge_Graph_idx]]
            ), axis=1, dtype=linear_type))

    # Graph (cluster) analysis
    clusters = G.components()
    membership = np.array(clusters.membership, dtype=linear_type)
    cluster_ids = np.argsort(clusters.sizes())[::-1]
    G.clear(); del G; del clusters
    
    # Loop through clusters of free volume spheres in sorted order largest to smallest
    for i,id in enumerate(cluster_ids):
        # Only analyze the largest free volume sphere (radius r >= probe_radius) cluster, i.e., i = 0
        if args.solvent_name == 'percolated':
            if i == 0:
                # Remove all free volume voxels not within the largest cluster, i.e., id(i == 0)
                radii_arr[(membership != id) & (radii_arr >= args.probe_radius)] = args.probe_radius/2                                          # Set radii = probe_radius/2 so that these voxels are treated as free volume VOXELS and not free volume SPHERES going forward
                break
        # Only analyze free volume sphere (radius r >= probe_radius) clusters which contain solvent
        else:
            clust = np.where(membership == id)[0]                                                                                               # Linearized indices

            # Clusters containing 1 free volume sphere are assumed to NOT contain solvent
            #  - Significantly reduces compute time
            if len(clust) == 1:
                radii_arr[np.isin(membership, cluster_ids[i:]) & (radii_arr >= args.probe_radius)] = args.probe_radius/2                        # Set radii = probe_radius/2 so that these voxels are treated as free volume VOXELS and not free volume SPHERES going forward
                break

            # For efficiency, we limit the number of free volume spheres per loop to a total of N_calc_max distance calculations
            count = 0
            while count < len(clust):
                count_old = count; count += min(int(args.N_calc_max/N_sol), len(clust)-count_old)

                # Useful print command for troubleshooting memory problems
                # Decreasing N_calc_max will reduce memory usage
                if (args.print_eff == 2) and (last_frame or args.N_threads == 1) and (i == 0 or len(clust)*N_sol > args.N_calc_max/10):
                    if count_old == 0: print(f"\nCluster size: {len(clust)}")
                    print(f"Calculations: {(count - count_old)*N_sol:.1e}")
                
                idx_x, idx_y, idx_z = np.unravel_index(clust[count_old:count], (l_x, l_y, l_z))                                                 # Spatial indices
                # Find the number of solvent atoms within probe_radius of a free volume sphere center
                pair_arr = distances.capped_distance(np.stack((vox_x[idx_x], vox_y[idx_y], vox_z[idx_z]),axis=1, dtype=FLOAT_TYPE), sol, args.probe_radius, box=cell, return_distances=False)
    
                # If any solvent molecules are found within the free volume sphere cluster, analysis can end early
                if len(pair_arr) != 0: break
    
            # If any solvent atoms are within probe_radius of a free volume sphere, then the entire cluster is considered a part of the solvent domain
            if len(pair_arr) != 0: continue

            # All other clusters are removed from the free volume sphere analysis.
            radii_arr[clust] = args.probe_radius/2  

    radii_arr = radii_arr.reshape((l_x, l_y, l_z))
    max_diameter = 2*np.max(radii_arr)

    # Useful print command for troubleshooting problems
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_Cluster = time.perf_counter() - time_Cluster
        print(f"\nMaximum pore diameter: {max_diameter:.2f}")
        if args.solvent_name == 'percolated':
            print(f"Number of spheres (r >= probe_radius) within percolated domain: {len(radii_arr[radii_arr >= args.probe_radius])}")
        else:
            print(f"Number of spheres (r >= probe_radius) within solvent domain: {len(radii_arr[radii_arr >= args.probe_radius])}")
        print(f"Time cluster: {time_Cluster:.2f} s\n")

    return radii_arr, max_diameter, time_Cluster