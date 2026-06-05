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

# This file contains functions to calculate the pore size distribution, fractional free volume, surface area, and tortuosity
# of the system based on the voxel-centered free volume spheres. These functions are called in the main function in parse.py.

import argparse
import MDAnalysis.lib.distances as distances
import numpy as np
import porespy as ps
from skimage import measure
import time
from typing import Tuple, Dict, Any, Optional, List

from utils.constants import FLOAT_TYPE

# suppresses expected warning in tortuosity analysis
ps.settings.loglevel = 'ERROR'

def calculate_psd_ffv(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        last_frame: bool, 
        cell: np.ndarray, 
        radii_arr: np.ndarray, 
        max_diameter: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculates Fractional Free Volume and Pore Size Distributions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        last_frame (bool): True if the current frame is the last frame in frame_ids.
        cell (np.ndarray): Array of simulation cell dimensions.
        radii_arr (np.ndarray): Radius of the largest free volume sphere centered on each voxel.
        max_diameter (float): Largest free volume sphere diameter.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: PSD bins, PSD data, FFV data, index of free volume voxels, index of free volume voxel diameters bins, and time to perform calculation
    """
    
    # Read in voxel_data
    vox_x, vox_y, vox_z = voxel_data['vox_x'], voxel_data['vox_y'], voxel_data['vox_z']
    l_x, l_y, l_z = voxel_data['l_x'], voxel_data['l_y'], voxel_data['l_z']
    indexed_type = voxel_data['indexed_type']
    
    d_arr = np.insert(np.arange(2*args.probe_radius, args.d_max + args.d_step, args.d_step),0,0); PSD_arr = np.zeros_like(d_arr, dtype=int)     # d_arr is the histogram of free volume sphere sizes; PSD_arr tracks the number of instances of voxels contained within free volume spheres of size at least d
    FFV_c = 0; FFV_lr = 0; FFV_total = 0                                                                                                        # Track number of voxels within the Connolly and Lee-Richards free volume against the total number to get FFV
    PSD_probes = np.indices((l_x, l_y, l_z), dtype=indexed_type).reshape(3, -1).T                                                               # Indices of all
    FFV_save = np.array([[],[],[]], dtype=indexed_type).T; d_save = np.array([], dtype=indexed_type)                                            # Save voxel-centers within the free volume for surface area calculations, and the size of the largest free volume sphere containing each voxel-center for printing in Free_Volume_Voxels.xyz

    # Ensure max_diameter > d_max
    if max_diameter > args.d_max:
        raise ValueError(f"Largest pore diameter is greater than d_max: {max_diameter} > {args.d_max}")

    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_PSD = time.perf_counter()
        print('\n##### Performing PSD/FFV Analysis #####\n')
    else: time_PSD = 0.0

    # Starting from the largest free volume spheres, find all free volume voxels within the desired free volume domain for FFV and PSD calculations
    cycle = 0; err = np.inf; N_rand = int(np.ceil((l_x * l_y * l_z) * args.rand_frac)); PSD_Old = np.zeros_like(PSD_arr); N_calc_max_temp = args.N_write_max
    while err > args.tol and len(PSD_probes) != 0:
        if N_rand > len(PSD_probes): N_rand = len(PSD_probes)

        # Track number of cycles
        cycle += 1
        if (args.print_eff == 2) and (last_frame or args.N_threads == 1):
            print(f"PSD Cycle: {cycle:5d}/{int(np.ceil(1/args.rand_frac))}")

        if args.rand_frac == 1.0:
            PSD_temp = PSD_probes[:]
            PSD_probes = np.array([])
        else:
            rng = np.random.default_rng()
            Rand_idx = rng.choice(len(PSD_probes), size=N_rand, replace=False)
            PSD_temp = PSD_probes[Rand_idx]
            PSD_probes = np.delete(PSD_probes, Rand_idx, axis=0)
            
        FFV_total += N_rand
        FFV_lr += np.sum(radii_arr[PSD_temp[:,0],PSD_temp[:,1],PSD_temp[:,2]] >= args.probe_radius)                                                 # The Lee-Richards volume is already known from the voxel-centered free volume spheres of radius r >= probe_radius
        PSD_temp = PSD_temp[radii_arr[PSD_temp[:,0], PSD_temp[:,1], PSD_temp[:,2]] != 0]                                                            # Remove voxels within the system domain from the PSD/FFV analysis

        # For efficiency, we measure the distance between free volume spheres and the voxel-centers starting with the largest d_arr bin and moving down
        for d in np.round(np.arange(args.d_max, 0, -args.d_step), decimals = 5):
            if d - args.d_step > max_diameter: continue
            if (d < 2*args.probe_radius) or (len(PSD_temp) == 0): break

            if (d - args.d_step)/2 < args.probe_radius:
                idx_x, idx_y, idx_z = np.where((radii_arr <= d/2) & (radii_arr >= args.probe_radius))
            else:
                idx_x, idx_y, idx_z = np.where((radii_arr <= d/2) & (radii_arr > (d - args.d_step)/2))
                
            sphere_temp = np.stack((vox_x[idx_x],vox_y[idx_y],vox_z[idx_z]), axis=1, dtype=FLOAT_TYPE); radii_temp = radii_arr[idx_x, idx_y, idx_z] # Positions (sphere_temp) and radii (radii_temp) of free volume spheres in the current PSD bin, radius (d - d_step)/2 < r <= d/2
            if len(sphere_temp) == 0: continue

            # For efficiency, we limit the number of free volume spheres per loop to a total of N_calc_max distance calculations
            count = 0; increase_max = False
            while count < len(sphere_temp) and len(PSD_temp) > 0:
                count_old = count; count += min(int(N_calc_max_temp/len(PSD_temp)), len(sphere_temp)-count_old)

                sph_temp = sphere_temp[count_old:count]; rad_temp = radii_temp[count_old:count]
                pair_arr, dist_arr = distances.capped_distance(sph_temp, np.stack((vox_x[PSD_temp[:,0]], vox_y[PSD_temp[:,1]], vox_z[PSD_temp[:,2]]), axis=1, dtype=FLOAT_TYPE), d/2 + 0.5, box=cell) # Distance between free volume spheres and voxel-centers

                # Control the number of writes
                if not increase_max and N_calc_max_temp < args.N_calc_max and count < len(sphere_temp) and len(dist_arr) < args.N_write_max/10: increase_max = True

                # Useful print command for troubleshooting memory problems
                # Decreasing N_calc_max will reduce memory usage
                if (args.print_eff == 2) and (last_frame or args.N_threads == 1) and ( (len(sph_temp)*len(PSD_temp) > args.N_calc_max/10) or (len(dist_arr) > args.N_write_max/10) ):
                    if count_old == 0: print(f"\nDiameter: {d - args.d_step} < d <= {d}")
                    print(f"Calculations, writes: {len(sph_temp)*len(PSD_temp):.1e} {len(dist_arr):.1e}")
                
                
                if len(dist_arr) > 0:
                    dist_arr -= rad_temp[pair_arr[:,0]]                                                                                         # Subtract radius of each free volume sphere from the distance to get the distance from the voxel-center to the surface of the free volume sphere
                    pair_arr = np.unique(pair_arr[:,1][dist_arr < 0])                                                                           # Only consider voxel-centers that lie within the free volume sphere (adjusted distance < 0), and only count each occurrence once (unique)

                    FFV_c += len(pair_arr); PSD_arr[np.where(d_arr < d)[0]] += len(pair_arr)                                                    # Voxel-centers w/n free volume sphere count towards the FFV and cumulatively towards the PSD

                    FFV_save = np.append(FFV_save, PSD_temp[pair_arr], axis=0)                                                                  # Save free volume voxel-centers for printing
                    d_save = np.append(d_save, np.zeros((len(pair_arr)), dtype=int) + int(d/args.d_step))

                    PSD_temp = np.delete(PSD_temp, pair_arr, axis=0)                                                                            # No longer consider voxel-centers that are found within a free volume sphere in future loops (prevent double-counting)
            if increase_max: N_calc_max_temp = min(args.N_calc_max, 10*N_calc_max_temp)

        PSD = PSD_arr / PSD_arr[0]; PSD = -(PSD[1:] - PSD[:-1]) / args.d_step
        if np.all(PSD_Old == 0):
            PSD_Old = PSD
            continue
        err = np.max(np.abs(np.divide((PSD - PSD_Old), PSD, out=np.zeros_like(PSD), where=(PSD != 0)))); PSD_Old = PSD

        if (args.print_eff == 2) and (last_frame or args.N_threads == 1):
            if args.tol > 0: print(f"Maximum Error/Tolerance: {err:.1e}/{args.tol:.1e}\n")
            else: print(f"Maximum Error: {err:.1e}\n")

    # Code to print the final FFV and PSD for the last frame analyzed
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_PSD = time.perf_counter() - time_PSD
        print(f"Connolly FFV: {FFV_c/FFV_total:0.3f}, {FFV_c}, {FFV_total}")
        print(f"Lee-Richards FFV: {FFV_lr/FFV_total:0.3f}, {FFV_lr}, {FFV_total}")
        print(f"\nPSD Final: {PSD_arr[0]}")
        print_string=''
        for i in PSD_arr:
            if i != 0: print_string += str(np.round(i / PSD_arr[0], decimals=5)) + ' '
        print(print_string)
        print(f"Time PSD/FFV: {time_PSD:.2f} s")

    return d_arr, PSD_arr, np.array([FFV_c, FFV_lr, FFV_total], dtype=int), FFV_save, d_save, time_PSD


def calculate_surface_area(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        last_frame: bool, 
        radii_arr: np.ndarray, 
        FFV_save: np.ndarray) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], np.ndarray], float]:
    """
    Calculates Connolly and Lee-Richards surface areas, returning surface meshes.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        last_frame (bool): True if the current frame is the last frame in frame_ids.
        radii_arr (np.ndarray): Radius of the largest free volume sphere centered on each voxel.
        FFV_save (np.ndarray): Index of free volume voxels.
        
    Returns:
        Tuple[np.ndarray, Tuple[Optional[np.ndarray], np.ndarray], float]: SA data, surface mesh data, and time to perform calculation.
    """
    
    # Read in voxel_data
    L_voxel_x, L_voxel_y, L_voxel_z = voxel_data['L_voxel_x'], voxel_data['L_voxel_y'], voxel_data['L_voxel_z']
    l_x, l_y, l_z = voxel_data['l_x'], voxel_data['l_y'], voxel_data['l_z']
    
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_SA = time.perf_counter()
        print('\n##### Performing SA Analysis #####\n')
    else: time_SA = 0.0

    ######################################################
    ############### Connolly Surface Area ################
    ######################################################
    if args.PSD_FFV:
        SA_arr = np.zeros((l_x, l_y, l_z), dtype=bool); SA_arr[FFV_save[:,0], FFV_save[:,1], FFV_save[:,2]] = True                              # Create voxel lattice where free volume voxel-centers = True

        # Create a simple mesh surface around the free volume and calculate the surface area
        SA_arr = np.pad(SA_arr, pad_width = 1, mode = 'wrap')                                                                                   # Add 1 layer of wrapped coordinates around the array to properly account for periodic boundaries
        spacing = np.array([L_voxel_x, L_voxel_y, L_voxel_z])                                                                                   # Define voxel size to dimensionalize surface area calculations

        verts_c, faces_c, _, _ = measure.marching_cubes(SA_arr, level = 0.5, spacing = spacing)                                                 # Marching cubes algorithm to create a surface mesh
        SA_c = measure.mesh_surface_area(verts_c, faces_c)                                                                                      # Calculate the surface area of the free volume
    else:
        SA_c = -1; verts_c = None

    ######################################################
    ### Lee-Richards "Surface Accessible" Surface Area ###
    ######################################################
    # Surface defined by the *center* of the free volume *spheres* - i.e., surface-accessible free volume
    idx_x, idx_y, idx_z = np.where(radii_arr >= args.probe_radius)
    SA_arr = np.zeros((l_x, l_y, l_z), dtype=bool); SA_arr[idx_x, idx_y, idx_z] = True                                                          # Create voxel lattice where free volume sphere-centers = True

    # Create a simple mesh surface around the free volume and calculate the surface area
    SA_arr = np.pad(SA_arr, pad_width = 1, mode = 'wrap')                                                                                       # Add 1 layer of wrapped coordinates around the array to properly account for periodic boundaries
    spacing = np.array([L_voxel_x, L_voxel_y, L_voxel_z])                                                                                       # Define voxel size to dimensionalize surface area calculations

    verts_lr, faces_lr, _, _ = measure.marching_cubes(SA_arr, level = 0.5, spacing = spacing)                                                   # Marching cubes algorithm to create a surface mesh
    SA_lr = measure.mesh_surface_area(verts_lr, faces_lr)                                                                                       # Calculate the surface area of the free volume

    # Normalize surface area to the true volume to account for padding
    volume = (l_x * L_voxel_x) * (l_y * L_voxel_y) * (l_z * L_voxel_z)
    padded_volume = ((l_x + 2) * L_voxel_x) * ((l_y + 2) * L_voxel_y) * ((l_z + 2) * L_voxel_z)
    if args.PSD_FFV: SA_c *= (volume / padded_volume)
    SA_lr *= (volume / padded_volume)

    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_SA = time.perf_counter() - time_SA
        print(f"Connolly SA (A^2):  {SA_c:.2f}")
        print(f"Lee-Richards SA (A^2):  {SA_lr:.2f}")
        print(f"Time SA: {time_SA:.2f} s")

    return np.array([SA_c, SA_lr], dtype=float), (verts_c, verts_lr), time_SA


def calculate_tortuosity(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        last_frame: bool, 
        radii_arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates Tortuosity using PoreSpy along X, Y, and Z axes.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        last_frame (bool): True if the current frame is the last frame in frame_ids.
        radii_arr (np.ndarray): Radius of the largest free volume sphere centered on each voxel.
        
    Returns:
        Tuple[np.ndarray, float]: Tau data and time to perform calculation.
    """
    
    # Read in voxel_data
    l_x, l_y, l_z = voxel_data['l_x'], voxel_data['l_y'], voxel_data['l_z']
    
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_tau = time.perf_counter()
        print('\n##### Performing Tortuosity Analysis #####\n')
    else: time_tau = 0.0

    # Diffusive volume is defined by *probe-center* occupiable volume, i.e., the Lee-Richards volume
    idx_x, idx_y, idx_z = np.where(radii_arr >= args.probe_radius)
    tortuosity_arr = np.zeros((l_x, l_y, l_z), dtype=bool); tortuosity_arr[idx_x, idx_y, idx_z] = True                                          # Create voxel lattice where free volume sphere-centers = True

    try:                                                                                                                                        # Attempt tortuosity analysis across x, y, and z directions
        sim_x = ps.simulations.tortuosity_fd(tortuosity_arr, axis=0); tortuosity_x = sim_x.tortuosity                                           # Analysis fails if no percolating clusters found across that axis
        sim_y = ps.simulations.tortuosity_fd(tortuosity_arr, axis=1); tortuosity_y = sim_y.tortuosity
        sim_z = ps.simulations.tortuosity_fd(tortuosity_arr, axis=2); tortuosity_z = sim_z.tortuosity
    except Exception as e:
        if "No pores remain" in str(e):                                                                                                         # If no percolating cluster found across any axis, return -1 for failed analysis
            if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
                print("Warning: Void space does not percolate along at least one axis. Setting tortuosity to -1.")
            tortuosity_x = -1; tortuosity_y = -1; tortuosity_z = -1
        elif "Solver failed to converge" in str(e):                                                                                             # If solver failed to converge across any axis, return -1 for failed analysis
            if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
                print("Error: Solver failed to converge along at least one axis. Setting tortuosity to -1.")
            tortuosity_x = -1; tortuosity_y = -1; tortuosity_z = -1
        else: raise e

    tortuosity = np.mean([tortuosity_x, tortuosity_y, tortuosity_z])                                                                            # Average tortuosity across all 3 dimensions

    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        time_tau = time.perf_counter() - time_tau
        if tortuosity == -1: print(f"No 1D percolated clusters found, tortuosity not measured.")
        else:
            print(f"Directional Tortuosity:  X-{tortuosity_x:.2f} Y-{tortuosity_y:.2f} Z-{tortuosity_z:.2f}")
            print(f"Average Tortuosity:  {tortuosity:.2f}")
        print(f"Time Tortuosity: {time_tau:.2f} s")

    return np.array([tortuosity_x, tortuosity_y, tortuosity_z], dtype=float), time_tau