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

# Voxelization functions

import argparse
import MDAnalysis.lib.distances as distances
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional, List

from utils.constants import FLOAT_TYPE

######################################################################
####################### Analysis Functions ###########################
######################################################################

def voxelize_system(
        args: argparse.Namespace, 
        cell: np.ndarray) -> Dict[str, Any]:
    """
    Generates and returns the initial coordinate grids and data types.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cell (np.ndarray): Array of simulation cell dimensions.
        
    Returns:
        Dict[str, Any]: Dictionary containing grid coordinates and specific numpy datatypes.
    """

    vox_x = np.linspace(0, cell[0], num = np.ceil(cell[0]/args.L_voxel).astype(int), dtype=FLOAT_TYPE); vox_x = (vox_x[:-1] + vox_x[1:])/2      # Voxel-centers in the x direction
    vox_y = np.linspace(0, cell[1], num = np.ceil(cell[1]/args.L_voxel).astype(int), dtype=FLOAT_TYPE); vox_y = (vox_y[:-1] + vox_y[1:])/2      # Voxel-centers in the y direction
    vox_z = np.linspace(0, cell[2], num = np.ceil(cell[2]/args.L_voxel).astype(int), dtype=FLOAT_TYPE); vox_z = (vox_z[:-1] + vox_z[1:])/2      # Voxel-centers in the z direction

    # True L_voxel in x, y, and z
    L_voxel_x = vox_x[1] - vox_x[0]; L_voxel_y = vox_y[1] - vox_y[0]; L_voxel_z = vox_z[1] - vox_z[0]
    # Box lengths in units of number of voxels
    l_x = len(vox_x); l_y = len(vox_y); l_z = len(vox_z)

    if args.Voxel_dist == 'Random':
        rng = np.random.default_rng()
        # Add random offsets to break up the uniformity of the voxels
        vox_x = vox_x + rng.uniform(low=-L_voxel_x/2, high=L_voxel_x/2, size=vox_x.size)
        vox_y = vox_y + rng.uniform(low=-L_voxel_y/2, high=L_voxel_y/2, size=vox_y.size)
        vox_z = vox_z + rng.uniform(low=-L_voxel_z/2, high=L_voxel_z/2, size=vox_z.size)
        
    # Use smallest integer data types possible (without losing precision) to reduce memory usage
    indexed_type = np.min_scalar_type(np.max([l_x, l_y, l_z]) - 1)
    linear_type = np.min_scalar_type((l_x * l_y * l_z) - 1)
    signed_linear_type = np.min_scalar_type(-((l_x * l_y * l_z) - 1))

    return {
        'vox_x': vox_x, 'vox_y': vox_y, 'vox_z': vox_z,
        'L_voxel_x': L_voxel_x, 'L_voxel_y': L_voxel_y, 'L_voxel_z': L_voxel_z,
        'l_x': l_x, 'l_y': l_y, 'l_z': l_z,
        'indexed_type': indexed_type, 'linear_type': linear_type, 'signed_linear_type': signed_linear_type
    }


def generate_free_volume_spheres(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        last_frame: bool, 
        cell: np.ndarray, 
        sys: np.ndarray, 
        sys_radii: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Find the largest free volume sphere centered on each voxel.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        last_frame (bool): True if the current frame is the last frame in frame_ids.
        cell (np.ndarray): Array of simulation cell dimensions.
        sys (np.ndarray): Coordinates of all system atoms.
        sys_radii (np.ndarray): vdW radius of all system atoms.
        
    Returns:
        Tuple[np.ndarray, float, float]: Array of radii of the largest free volume sphere centered on each voxel, maximum diameter, and time to perform calculation.
    """
    
    # Read in voxel_data
    vox_x, vox_y, vox_z = voxel_data['vox_x'], voxel_data['vox_y'], voxel_data['vox_z']
    L_voxel_x, L_voxel_y, L_voxel_z = voxel_data['L_voxel_x'], voxel_data['L_voxel_y'], voxel_data['L_voxel_z']
    l_x, l_y, l_z = voxel_data['l_x'], voxel_data['l_y'], voxel_data['l_z']
    indexed_type = voxel_data['indexed_type']
    
    # Efficiency parameters used to determine N_cube
    avg_sys_density = len(sys) / cell[0] / cell[1] / cell[2]
    vol_d_inc = (4/3) * np.pi * (args.d_inc**3)

    # --d_inc must be a minimum value to prevent an error in generating the free volume spheres
    if args.d_inc < np.max(sys_radii) - np.min(sys_radii):
        raise ValueError(f"Set --d_inc to {np.max(sys_radii) - np.min(sys_radii):.2f} or larger")
    # In the niche case where all system atoms have the same vdW radius, this is not an issue
    skip_dinc_check = (np.max(sys_radii) - np.min(sys_radii) == 0)

    radii_arr = np.zeros((l_x,l_y,l_z), dtype=FLOAT_TYPE)                                                                                       # radii_arr tracks free volume sphere indices (position in array = position in voxelized system) and radius (value at that position), where we are interested in spheres of radius r >= probe_radius. All probes of r > 0 are saved for later use.

    # Divide the voxelized system into voxel cubes for efficient analysis
    N_cube = np.min([                                                                                                                           # To improve efficiency, voxels are looped over in cubes of N_cube voxel-centers
        args.N_write_max / (avg_sys_density * vol_d_inc),                                                                                       # N_cube to achieve approx. N_write_max
        (args.N_calc_max / (avg_sys_density * (args.L_voxel + args.d_inc/2)**3)) ** (1/2)                                                       # N_cube to achieve approx. N_calc_max
    ])                                                                                                                                          #   Choose the smallest value to ensure neither limit is breached
    L_cube = N_cube**(1/3)
    vox_inc = np.ceil(np.min((l_x, l_y, l_z)) / np.ceil(np.min((l_x, l_y, l_z)) / L_cube)).astype(int)                                          # vox_inc = side length of voxel cube, such that each voxel cube is about the same size
    N_cube = vox_inc**3                                                                                                                         # Actual number of voxels in each voxel cube after making each cube approximately the same size
    vox_track = np.array((-vox_inc,0,0), dtype=int)                                                                                             # vox_track tracks the location of the cubes in x, y, and z compared to the position in vox_x, vox_y, and vox_z

    # Prints the number of voxels-per-cube and number of voxel cubes
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_Spheres = time.perf_counter()
        print('##### Generating Free Volume Spheres #####')
        print(f"\nNumber of voxels-per-cube: {N_cube}")
        print(f"Number of voxel cubes: {np.ceil(l_x/vox_inc).astype(int)*np.ceil(l_y/vox_inc).astype(int)*np.ceil(l_z/vox_inc).astype(int)}")
    else: time_Spheres = 0.0

    for x_i in np.arange(vox_inc,l_x+vox_inc,vox_inc):
        vox_track[0] += vox_inc
        if x_i > l_x: x_i = l_x

        vox_track[1] = -vox_inc
        for y_i in np.arange(vox_inc,l_y+vox_inc,vox_inc):
            vox_track[1] += vox_inc
            if y_i > l_y: y_i = l_y

            vox_track[2] = -vox_inc
            for z_i in np.arange(vox_inc,l_z+vox_inc,vox_inc):
                vox_track[2] += vox_inc
                if z_i > l_z: z_i = l_z

                sphere_temp = np.vstack(np.meshgrid(                                                                                            # sphere_temp contains the position of the voxel-centers within the cube of size N_cube
                    vox_x[vox_track[0]:x_i], vox_y[vox_track[1]:y_i], vox_z[vox_track[2]:z_i]
                ), dtype=FLOAT_TYPE).reshape(3,-1).T

                # Find the approximate center of the voxel cube to find the system atoms near the voxel cube (sys_mask), where system atoms define the van der Waals volume of the system. Reduces computational cost
                center = np.array([
                    vox_x[vox_track[0] + int((x_i - vox_track[0])/2)],
                    vox_y[vox_track[1] + int((y_i - vox_track[1])/2)],
                    vox_z[vox_track[2] + int((z_i - vox_track[2])/2)]
                ], dtype=FLOAT_TYPE)

                # To reduce the number of calculations and limit memory usage, the distance between voxel-centers and system atoms is done in steps of d_inc Angstroms
                d = 0.0                                                                                                                         # Maximum distance to calculate between every voxel-center and every system atom
                while len(sphere_temp) > 0:
                    d += args.d_inc

                    sys_mask = distances.capped_distance(center, sys, d + np.sqrt(3)*vox_inc*args.L_voxel/2 + 2*args.L_voxel, box=cell, return_distances=False)[:,1] # System atoms near the voxel cube
                    pair_arr, dist_arr = distances.capped_distance(sphere_temp, sys[sys_mask], d, box=cell)                                     # Distance between voxel-centers and system atoms

                    # Useful print command for troubleshooting memory problems
                    # Decreasing N_write_max/N_calc_maX will reduce the number of distances generated each cycle, reducing memory usage
                    if (args.print_eff == 2) and (last_frame or args.N_threads == 1):
                        if d == args.d_inc: print(f"\nVoxel block: {(vox_track/vox_inc).astype(int)}")
                        print(f"Distance, calculations, writes: {d:3.1f} {len(sphere_temp)*len(sys[sys_mask]):.1e} {len(dist_arr):.1e}")

                    if len(dist_arr) > 0:
                        dist_arr -= sys_radii[sys_mask][pair_arr[:,1]]                                                                          # Subtract radius of each system atom from the distance to get the distance from the voxel-center to the surface of the atom

                        # Fill radii_arr for all voxel-centers that contain system atoms within d distance, where the smallest distance is the radius of the free volume sphere centered on the voxel
                        index = 0; sph_save = pair_arr[0,0]; sphere_remove = []
                        for i,sph in enumerate(pair_arr[:,0]):
                            if (sph > sph_save) or (i+1 == len(pair_arr[:,0])):
                                r_min = np.min(dist_arr[index:i])                                                                               # Minimum distance between voxel-center and system surface

                                remove_sph = False                                                                                              # Only remove from future calculations if r_min < 0 or r_min is the same value for two cycles. This prevents artificially large radii due to the incremental (d_inc) algorithm.
                                if r_min > 0:                                                                                                   # Sphere does not overlap the system and radius >= 0
                                    coords = np.divide(sphere_temp[sph_save], np.array([L_voxel_x, L_voxel_y, L_voxel_z])).astype(indexed_type)
                                    if radii_arr[coords[0],coords[1],coords[2]] == r_min: remove_sph = True
                                    radii_arr[coords[0],coords[1],coords[2]] = r_min
                                else: remove_sph = True

                                if skip_dinc_check or remove_sph: sphere_remove.append(sph_save)                                                 # Analysis complete, remove from future distance calculations
                                index = i; sph_save = sph
                    if len(sphere_remove) > 0: sphere_temp = np.delete(sphere_temp, np.array(sphere_remove), axis=0)                             # Remove evaluated voxel-centers

    max_diameter = 2 * np.max(radii_arr)

    # Useful print command for troubleshooting problems: prints the number of voxel-centers within the free volume and the diameter of the largest sphere (pore)
    # Also prints the number of voxels within the system van der Waals free volume, voxels containing free volume spheres of radius r >= probe_radius, and voxels that need to be assessed whether they are in the free volume or not
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_Spheres = time.perf_counter() - time_Spheres
        print(f"\nMaximum pore diameter: {max_diameter:.2f}")
        print(f"Number of free volume spheres (r >= probe_radius): {len(radii_arr[radii_arr >= args.probe_radius])}")
        print(f"Number of free volume voxels (r > 0): {len(radii_arr[radii_arr != 0])}")
        print(f"Time free volume spheres: {time_Spheres:.2f} s")

    return radii_arr, max_diameter, time_Spheres