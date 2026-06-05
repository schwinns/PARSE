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

# PARSE: Pore Analysis & Reconstruction of Spatial Environments
# PARSE.py calculates the pore size distribution (free volume distribution, channel width distribution, etc) of the van der Waals free volume of the defined system matrix from a GROMACS (xtc/trr/gro + tpr/gro) or PoreBlazer-style (xyz + dat) trajectory
# This script was specifically designed to find the distribution of water-rich pores within a hydrated polymer system, but can be generalized to any atomic or coarse-grained system
# The output includes the Cumulative Pore Size Distribution (Cumulative PSD), Pore Size Distribution (PSD), and Free Volume Fraction (Fractional Free Volume, FFV), with optional Surface Area (SA), Tortuosity (Tau), and xyz visualizations
# Given the system matrix (e.g., polymer matrix, non-polar domain, polar domain, etc), this code can analyze the total free volume, free volume of the largest cluster (assumed percolated), or the free volume of clusters containing (defined) solvent atoms
# This script was written based on the methods used for PoreBlazer (https://github.com/SarkisovGitHub/PoreBlazer) and is optimized for parallelized calculations over many system frames, or analysis of large (30+ nm box length) systems
#
# As written, this code reads in GROMACS trajectory or PoreBlazer xyz and dat data using MDAnalysis
# As written, this code is designed for 3D-periodic rectangular simulations
#
# When implementing this code, it is recommended to test different values of L_voxel to ensure convergence of the FFV as L_voxel decreases. Note, computation time and memory usage will grow significantly as L_voxel decreases.
# If you run into memory or extreme run times, there are debugging lines throughout the code, and several values you can change to increase or decrease memory usage.
# There are three instances where xyz files can be created to visualize 1) probe-occupiable spheres of maximum radius without overlapping the van der Waals volume of the system, 2) voxel-centers that lie within the probe-occupiable volume, and 3) voxel-center surfaces that define the Connolly or Lee-Richards surface of the probe-occupiable volume

import numpy as np
import h5py

import multiprocessing as mp
import functools
import os
import sys
import time
import argparse
from typing import Dict, Any

# Import analysis functions
from core.analysis import calculate_psd_ffv, calculate_surface_area, calculate_tortuosity
from core.voxels import voxelize_system, generate_free_volume_spheres
from core.cluster import perform_clustering_analysis

# Import utility functions
from utils.parsing import load_Args
from utils.files import load_Trajectory
from utils.files import write_FFV, write_PSD, write_Surface_area, write_Tortuosity
from utils.files import export_spheres_xyz, export_voxels_xyz, export_surface_xyz

######################################################################
####################### Analysis  Pipeline ###########################
######################################################################

def volume_analysis(
        args: argparse.Namespace, 
        frame_idx: int) -> Dict[str, Any]:
    """
    Orchestrates the volume analysis pipeline for a single frame.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        frame_idx (int): Index of current frame_idx.
        
    Returns:
        Dict[str, Any]: Dictionary containing PSD, FFV, SA, and tortuosity data for the current frame.
    """

    # Sleep command to offset processes (limit spikes in memory usage) - no delay if N_threads = 1
    rng = np.random.default_rng()
    time.sleep(frame_idx%args.N_threads)

    with h5py.File('PARSE.hdf5','r') as f:
        frame_ids = f['frames'][:]; frame = frame_ids[frame_idx]
        sys = f['system'][frame]                                                                                                                # Position of all system atoms
        sys_radii = f['sys_radii'][:]                                                                                                           # van der Waals radii of all system atoms
        sol = f['solvent'][frame]                                                                                                               # Position of all solvent atoms
        cell = f['cells'][frame]                                                                                                                # Size of the cell
    if frame_idx == len(frame_ids) - 1: last_frame = True
    else:                               last_frame = False

    # Track which frames are currently being processed
    print(f"Frame {frame_idx + 1}/{len(frame_ids)}")

    # Voxelize the system
    voxel_data = voxelize_system(args, cell)

    # Free Volume Sphere Analysis
    #   For each voxel, find the largest voxel-centered free volume sphere without overlapping system atoms (the van der Waals volume), where the total volume of all spheres larger than probe_radius defines the probe-occupiable free volume of the system
    #   Changing L_voxel, N_write_max/N_calc_max, and d_inc can reduce run time and memory usage
    radii_arr, max_diameter, time_Spheres = generate_free_volume_spheres(
        args, voxel_data, last_frame, cell, sys, sys_radii
    )

    # Clustering Analysis
    #   Only consider free volume spheres of radius r >= probe_radius that are within the desired domain.
    #   Free volume spheres outside of the desired domain are demoted from free volume spheres of radius r >= probe_radius to free volume voxels (0 < r < probe_radius), where such voxels are still considered in the FFV and PSD analysis.
    #   NOTE: This section is the most sensitive to memory errors. Consider setting solvent_name = "" if consistently running out of memory (OOM).
    if (args.solvent_name == 'percolated') or (len(sol) > 0):
        radii_arr, max_diameter, time_Cluster = perform_clustering_analysis(
            args, voxel_data, last_frame, cell, radii_arr, sol
        )
    else: time_Cluster = 0
    
    # Export free volume spheres to .xyz file for visualization
    if args.print_xyz and last_frame:
        export_spheres_xyz(args, voxel_data, radii_arr)

    # PSD/FFV Analysis
    #   Calculate the free volume fraction and cumulative probe-occupiable pore size distribution, where the distribution is defined as the probability that a random point (voxel) within the free volume resides within a free volume sphere of diameter d with minimum size probe_radius
    #   This code will take each voxel not within the system volume (PSD_probes) and determine 1) if it lies within the free volume (FFV), and 2) the largest free volume sphere it lies within (PSD)
    #   Changing L_voxel, N_calc_max, and d_step can reduce run time and memory usage
    if args.PSD_FFV:
        d_arr, PSD_arr, FFV_data, FFV_save, d_save, time_PSD = calculate_psd_ffv(
            args, voxel_data, last_frame, cell, radii_arr, max_diameter
        )
        # Export free volume voxels to .xyz file for visualization
        if args.print_xyz and last_frame:
            export_voxels_xyz(args, voxel_data, d_arr, FFV_save, d_save)
    else:
        d_arr = np.insert(np.arange(2*args.probe_radius, args.d_max + args.d_step, args.d_step), 0, 0); PSD_arr = np.zeros_like(d_arr, dtype=int) - 1

        FFV_c = -len(radii_arr.ravel()); FFV_lr = len(radii_arr[radii_arr >= args.probe_radius]); FFV_total = len(radii_arr.ravel())            # The Lee-Richards volume is already known from the voxel-centered free volume spheres of radius r >= probe_radius
        FFV_data = np.array([FFV_c, FFV_lr, FFV_total], dtype=int)

        FFV_save, time_PSD = None, 0
        if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
            print(f"Lee-Richards FFV: {FFV_lr/FFV_total:0.3f}, {FFV_lr}, {FFV_total}")

    # SA Analysis
    #   Find the Connolly and Lee-Richards surface using a marching cubes algorithm and calculate the surface area
    if args.Surface_area:
        SA_data, surface_meshes, time_SA = calculate_surface_area(
            args, voxel_data, last_frame, radii_arr, FFV_save
        )
        # Export free volume surface to .xyz file for visualization
        if args.print_xyz and last_frame:
            export_surface_xyz(args, cell, surface_meshes[0], surface_meshes[1])
    else: SA_data, time_SA = np.array([0,0], dtype=float), 0

    # Tau Analysis
    #   Calculate the tortuosity of the Lee-Richards volume using PoreSpy
    if args.Tortuosity:
        tortuosity_data, time_tau = calculate_tortuosity(
            args, voxel_data, last_frame, radii_arr
        )
    else: tortuosity_data, time_tau = np.array([0, 0, 0], dtype=float), 0

    # Print time statistics
    if (args.print_eff >= 1) and (last_frame or args.N_threads == 1):
        print("\n##### Summary of Calculation Times #####\n")
        print(f"Time free volume spheres: {time_Spheres:.2f} s")
        if (args.solvent_name == 'percolated') or (len(sol) > 0): print(f"Time cluster: {time_Cluster:.2f} s")
        if args.PSD_FFV: print(f"Time PSD/FFV: {time_PSD:.2f} s")
        if args.Surface_area: print(f"Time SA: {time_SA:.2f} s")
        if args.Tortuosity: print(f"Time Tortuosity: {time_tau:.2f} s")
    
    return {
        'PSD_arr': PSD_arr,
        'FFV': FFV_data,
        'SA': SA_data,
        'tortuosity': tortuosity_data
    }


def main():
    # Read in inputs from YAML file and command line
    args, Size_arr, Dummy_atoms, mda_kwargs = load_Args()

    print('########################################')
    print('########### Input Parameters ###########')
    print('########################################\n')
    for key, value in vars(args).items():
        if 'mode' in key:          print(  '    ############# Mode #############')
        elif 'trj_file' in key:    print('\n    ### Files and Run Parameters ###')
        elif 'system_name' in key: print('\n    ######## System/Solvent ########')
        elif 'L_voxel' in key:     print('\n    ########## Variables ###########')
        elif 'clustering' in key:  print('\n    #### Efficiency Parameters #####')

        if '_calc' in key or '_write' in key or 'target_' in key or '_gen' in key: print(f"    {key:18}: {value:.0e}")
        else:                                                                      print(f"    {key:18}: {value}")
    print('\n########################################')
    print(  '########################################')
    print(  '########################################\n')

    # Load in the trajectory file, save necessary data into h5py .hdf5 I/O file, and exit the code to purge the memory before multiprocessing
    # Must run the script a second time to perform the analysis
    if not os.path.exists('PARSE.hdf5'):
        print('Loading trajectory data\n')
        try:
            load_Trajectory(args, Size_arr, Dummy_atoms, mda_kwargs)
        except ValueError as e:
            print(f"ERROR - {e}")
            sys.exit(1)
        print('\nTrajectory loaded, terminating process. Run again to perform analysis')
        sys.exit(0)

    with h5py.File('PARSE.hdf5','r') as f:
        frame_ids = f['frames'][:]
    
    # If N_threads > N_frames * N_repeats, N_threads = N_frames * N_repeats
    if args.N_threads > len(frame_ids):
        print("--N_threads > args.N_frames * args.N_repeats, setting --N_threads (--N_frames * --N_repeats)")
        args.N_threads = len(frame_ids)

    # Perform the analysis using multiprocessing
    print("Volume Analysis\n")
    try:
        if args.N_threads == 1:
            out_arr = []
            for frame in frame_ids:
                out_arr.append(volume_analysis(args, frame))
        else:
            func = functools.partial(volume_analysis, args)
            with mp.Pool(processes=args.N_threads) as pool:
                out_arr = pool.map(func, range(len(frame_ids)))
    except ValueError as e:
        print(f"ERROR - {e}")
        sys.exit(1)
    
    # Write .dat files
    if args.PSD_FFV:
        write_PSD(args, np.array([out['PSD_arr'] for out in out_arr]))
        write_FFV(args, np.array([out['FFV'] for out in out_arr]))

    if args.Surface_area:
        write_Surface_area(np.array([out['SA'] for out in out_arr]))

    if args.Tortuosity:
        write_Tortuosity(np.array([out['tortuosity'] for out in out_arr]))
        
    # Deletes the temporary .hdf5 file
    os.remove('PARSE.hdf5')

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
