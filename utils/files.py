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

# Utility functions for file handling.

import argparse
import h5py
import MDAnalysis as mda
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import sys

from .constants import FLOAT_TYPE

######################################################################
######################## Trajectory Loading ##########################
######################################################################

def load_Trajectory(
        args: argparse.Namespace, 
        Size_arr: np.ndarray,
        Dummy_atoms: np.ndarray, 
        mda_kwargs: dict) -> None:
    """
    Loads in the trajectory and saves the necessary data to a temporary h5py .hdf5 I/O file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        Size_arr (np.ndarray): List of atoms and vdW radii.
        Dummy_atoms (np.ndarray): List of dummy atom names.
        mda_kwargs (dict): MDAnalysis Universe **kwargs input.
    """

    # Load in MDAnalysis Universe
    if args.mode == 'xyz':
        try:
            uta = mda.Universe(args.trj_file, **mda_kwargs)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Define the simulation cell from the dat file
        cell = np.zeros(6); cell[3:] = 90.0
        with open(args.top_file, 'r') as file:
            lines = file.readlines()[1]
            cell[:3] += np.array(lines.split(), dtype=float)
        print('If the following is incorrect, check your dat file format')
        print(f'XYZ cell size (A): {cell[0]} {cell[1]} {cell[2]}\n')
    else:
        try:
            uta = mda.Universe(args.top_file, args.trj_file, **mda_kwargs)
        except Exception as e:
            print(e)
            sys.exit(1)

    if len(uta.trajectory) < args.N_frames: raise ValueError(f"Requested more frames than available in the trajectory - Try --N_frames {len(uta.trajectory)}")

    # Define the system and solvent atoms
    system = uta.select_atoms(args.system_name)
    if args.solvent_name == 'percolated' or args.solvent_name == '': solvent = uta.select_atoms('not all')
    else: solvent = uta.select_atoms(args.solvent_name)

    print("If the following is incorrect, there may be inconsistencies between your atom ID name in the topology and the Element name in the YAML file (see 'Size_arr' and 'Dummy_atoms' for more details)")
    print("\nSYSTEM ATOMS")

    # If no system atoms are detected, return error
    if len(system) == 0: raise ValueError("No system atoms found")

    # Remove dummy atoms from the system
    if len(Dummy_atoms) > 0:
        if args.identify_atoms == 'Names':
            for dummy in Dummy_atoms:
                if np.sum(system.names == dummy) > 0:
                    print(f"Removed {np.sum(system.names == dummy)} {dummy} atoms from system analysis")
                    system = system[system.names != dummy]
        elif args.identify_atoms == 'Masses':
            if np.sum(system.masses == 0) > 0:
                print(f"Removed {np.sum(system.masses == 0)} dummy (massless) atoms from system analysis")
                system = system[system.masses != 0]
    
    # Create an array that tracks the radius of each system atom based on Size_array
    sys_radii = np.zeros((len(system)), dtype=FLOAT_TYPE); sys_count = np.zeros((len(Size_arr)), dtype=int)
    if args.identify_atoms == 'Names':
        sys_names = system.names
        for i, name in enumerate(sys_names):
            name = str(name)
            if name in Size_arr[:,0]:
                sys_radii[i] = float(Size_arr[np.where(Size_arr[:,0] == name)[0][0],2])
                sys_count[np.where(Size_arr[:,0] == name)[0][0]] += 1
            elif name[0] in Size_arr[:,0]:
                sys_radii[i] = float(Size_arr[np.where(Size_arr[:,0] == name[0])[0][0],2])
                sys_count[np.where(Size_arr[:,0] == name[0])[0][0]] += 1
            elif name == 'None': raise ValueError(f"Atom name == 'None', MDAnalysis names not assigned. Try --identify_atoms 'Masses'.")
            else: raise ValueError(f"Missing atom name and size in Size_arr: {name}")
    elif args.identify_atoms == 'Masses':
        sys_masses = system.masses; Mass_arr = np.round(Size_arr[:,1].astype(float))
        for i, mass in enumerate(sys_masses):
            mass = np.round(float(mass))
            if mass in Mass_arr:
                sys_radii[i] = float(Size_arr[np.where(Mass_arr == mass)[0][0],2])
                sys_count[np.where(Mass_arr == mass)[0][0]] += 1
            else: raise ValueError(f"Missing atom mass and size in Size_arr: {mass}")
    
    # --d_inc must be a minimum value to prevent an error in generating the free volume spheres
    if args.d_inc < np.max(sys_radii) - np.min(sys_radii):
        raise ValueError(f"Set --d_inc to {np.max(sys_radii) - np.min(sys_radii):.2f} or larger")

    # Print out system atom information
    print("Element N-in-System")
    for i,j in enumerate(sys_count):
        if j > 0: print(f"{Size_arr[i,0]:>7s} {j:11d}")
    
    if len(solvent) > 0:
        print("\nSOLVENT ATOMS")

        # Remove dummy atoms from the solvent
        if len(Dummy_atoms) > 0:
            if args.identify_atoms == 'Names':
                for dummy in Dummy_atoms:
                    if np.sum(solvent.names == dummy) > 0:
                        print(f"Removed {np.sum(solvent.names == dummy)} {dummy} atoms from solvent analysis")
                        solvent = solvent[solvent.names != dummy]
            elif args.identify_atoms == 'Masses':
                if np.sum(solvent.masses == 0) > 0:
                    print(f"Removed {np.sum(solvent.masses == 0)} dummy (massless) atoms from solvent analysis")
                    solvent = solvent[solvent.masses != 0]
        
        # Create an array that tracks the radius of each system atom based on Size_array
        sol_count = np.zeros((len(Size_arr) + 1), dtype=int)
        if args.identify_atoms == 'Names':
            sol_names = solvent.names
            for name in sol_names:
                name = str(name)
                if name in Size_arr[:,0]:
                    sol_count[np.where(Size_arr[:,0] == name)[0][0]] += 1
                elif name[0] in Size_arr[:,0]:
                    sol_count[np.where(Size_arr[:,0] == name[0])[0][0]] += 1
                else: sol_count[-1] += 1
        elif args.identify_atoms == 'Masses':
            sol_masses = solvent.masses
            for i, mass in enumerate(sol_masses):
                mass = np.round(float(mass))
                if mass in Mass_arr:
                    sol_count[np.where(Mass_arr == mass)[0][0]] += 1
                else: sol_count[-1] += 1

        # Print out solvent atom information
        print("Element N-in-System")
        for i,j in enumerate(sol_count):
            if j > 0:
                if i == len(sol_count) - 1:
                    print(f"{'Other':>7s} {j:11d}")
                    print("\nElement 'Other' means the atom is not defined as part of the van der Waals volume list.")
                    print("This is not an error and does not impact the code output.")
                else: print(f"{Size_arr[i,0]:>7s} {j:11d}")

    # Define the system times/frames to be calculated over
    print()
    if len(uta.trajectory) == 1:
        frame_ids = np.array([-1], dtype=int)
    else:
        if args.t_min == -1:    args.t_min    = uta.trajectory[0].time
        if args.t_max == -1:    args.t_max    = uta.trajectory[-1].time
        if args.N_frames == -1: args.N_frames = args.N_threads

        dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),3)

        start_idx = int((args.t_min - uta.trajectory[0].time) / dt)
        end_idx = int((args.t_max - uta.trajectory[0].time) / dt)
        available_frames = end_idx - start_idx + 1

        if available_frames < args.N_frames: raise ValueError(f"Not enough frames within the time range provided: {args.t_min}-{args.t_max} ps = {available_frames} frames")

        if args.N_frames == 1:                                                                  # If only analyzing one frame, analyze the final frame
            frame_ids = np.array([end_idx], dtype=int)
        else:
            frame_ids = np.linspace(start_idx, end_idx, args.N_frames, dtype=int)
            print(f"Timestep: ~{dt*(frame_ids[1] - frame_ids[0])} ps")
    print(f"Number of frames: {len(frame_ids)}")

    # Load in the necessary data: "system" atom positions, "solvent" atom positions, cell dimensions
    r_system = np.zeros((len(frame_ids), len(system), 3), dtype=FLOAT_TYPE)
    r_solvent = np.zeros((len(frame_ids), len(solvent), 3), dtype=FLOAT_TYPE)
    cells = np.zeros((len(frame_ids), 6))
    for i, frame in enumerate(frame_ids):
        ts = uta.trajectory[frame]

        r_system[i] = system.positions
        r_solvent[i] = solvent.positions

        if args.mode == 'xyz': cells[i] = cell
        else: cells[i] = ts.dimensions

        if np.any(cells[i,:3] / args.L_voxel < 1):      raise ValueError(f"--L_voxel is larger than at least one side of the simulation cell: {args.L_voxel} A, ({cells[i,0]:.3f},{cells[i,1]:.3f},{cells[i,2]:.3f}) A")
        if np.any(cells[i,:3] / args.probe_radius < 1): raise ValueError(f"--probe_radius is larger than at least one side of the simulation cell: {args.probe_radius} A, ({cells[i,0]:.3f},{cells[i,1]:.3f},{cells[i,2]:.3f}) A")

    
    # Convert frame_ids from index in the trajectory to index in the array and add repeats
    frame_ids = np.arange(0,len(frame_ids),1)
    if args.N_repeats > 1:
        frame_ids = np.repeat(frame_ids, args.N_repeats)
        print(f"Number of frames + repeats: {len(frame_ids)}")
    print(f"Number of threads: {args.N_threads}")

    # Save necessary information to a temporary .hdf5 file for later use in the calculation
    with h5py.File('PARSE.hdf5','w') as f:
        f.create_dataset("system", data=r_system, dtype=FLOAT_TYPE)
        f.create_dataset("sys_radii", data = sys_radii, dtype=FLOAT_TYPE)
        f.create_dataset("solvent", data=r_solvent, dtype=FLOAT_TYPE)
        f.create_dataset("cells", data = cells)
        f.create_dataset("frames", data = frame_ids)


######################################################################
######################## XYZ File Functions ##########################
######################################################################

def export_spheres_xyz(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        radii_arr: np.ndarray) -> None:
    """Exports the free volume spheres to a .xyz file for visualization.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        radii_arr (np.ndarray): Array of radii of the largest free volume spheres centered on each voxel.
    """

    # Read in voxel_data
    vox_x, vox_y, vox_z = voxel_data['vox_x'], voxel_data['vox_y'], voxel_data['vox_z']
    
    # Write .xyz file containing each free volume sphere of size r >= probe_radius
    #   Radius = radius of the free volume sphere
    idx_x, idx_y, idx_z = np.where(radii_arr >= args.probe_radius)
    with open('Free_Volume_Spheres.xyz', 'w') as anaout:
        print(str(len(idx_x)), file=anaout)
        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
        for i in range(len(idx_x)):
            x, y, z = vox_x[idx_x[i]], vox_y[idx_y[i]], vox_z[idx_z[i]]
            r = radii_arr[idx_x[i], idx_y[i], idx_z[i]]
            print(f"X {x:10.5f} {y:10.5f} {z:10.5f} {r:10.5f}", file=anaout)
    print('Free volume sphere xyz file printed')

def export_voxels_xyz(
        args: argparse.Namespace, 
        voxel_data: Dict[str, Any], 
        d_arr: np.ndarray, 
        FFV_save: np.ndarray, 
        d_save: np.ndarray) -> None:
    """Exports the free volume voxel centers to a .xyz file for visualization.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        voxel_data (Dict[str, Any]): Dictionary containing grid coordinates and specific numpy datatypes.
        d_arr (np.ndarray): PSD bins.
        FFV_save (np.ndarray): Index of free volume voxels.
        d_save (np.ndarray): Index of the diameter bin for the largest free volume sphere containing each free volume voxel.
    """

    # Read in voxel_data
    vox_x, vox_y, vox_z = voxel_data['vox_x'], voxel_data['vox_y'], voxel_data['vox_z']
    
    # Write .xyz file containing each free volume voxel
    #   Radius = L_voxel/2, Alpha = Diameter (largest diameter of that bin of d_arr) of the largest free volume sphere which contains the center of this voxel.
    with open('Free_Volume_Voxels.xyz', 'w') as anaout:
        print(str(len(FFV_save)), file=anaout)
        print('Properties=species:S:1:pos:R:3:Radius:R:1:Alpha:R:1', file=anaout)
        for i, sph in enumerate(FFV_save):
            x, y, z = vox_x[sph[0]], vox_y[sph[1]], vox_z[sph[2]]
            r = args.L_voxel/2; a = d_arr[d_save[i]]
            print(f"X {x:10.5f} {y:10.5f} {z:10.5f} {r:10.5f} {a:10.5f}", file=anaout)
    print('Free volume voxel xyz file printed')


def export_surface_xyz(
        args: argparse.Namespace, 
        cell: np.ndarray, 
        verts_c: Optional[np.ndarray], 
        verts_lr: np.ndarray) -> None:
    """Exports the surface mesh vertices to a .xyz file for visualization.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cell (np.ndarray): Array of simulation cell dimensions.
        verts_c (Optional[np.ndarray]): Surface mesh data for the Connolly surface.
        verts_lr (np.ndarray): Surface mesh data for the Lee-Richards surface.
    """

    # Remove excess voxels due to padding
    verts_c_save, verts_lr_save = [], []
    if args.PSD_FFV and verts_c is not None:
        for sph in verts_c:
            if np.any(sph < 0 - args.L_voxel/2) or np.any(sph > cell[:3] + args.L_voxel/2): continue
            verts_c_save.append(sph)
    for sph in verts_lr:
        if np.any(sph < 0 - args.L_voxel/2) or np.any(sph > cell[:3] + args.L_voxel/2): continue
        verts_lr_save.append(sph)

    # Write .xyz file containing each free volume voxel which makes up the Connolly and Lee-Richards surfaces
    # Voxels are centered on the surface of the free voxel volume
    #   X = Connolly surface, Y = Lee-Richards surface
    #   Radius = L_voxel/2
    with open('Free_Volume_Surface.xyz', 'w') as anaout:
        print(str(len(verts_c_save) + len(verts_lr_save)), file=anaout)
        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
        for sph in verts_c_save:
            print(f"X {sph[0]:10.5f} {sph[1]:10.5f} {sph[2]:10.5f} {args.L_voxel/2:10.5f}", file=anaout)
        for sph in verts_lr_save:
            print(f"Y {sph[0]:10.5f} {sph[1]:10.5f} {sph[2]:10.5f} {args.L_voxel/2:10.5f}", file=anaout)
    print('Free volume surface xyz file printed')



######################################################################
########################## Write Outputs #############################
######################################################################

def write_PSD(args: argparse.Namespace, PSD_arr: np.ndarray) -> None:
    """Writes the PSD and FFV data to .dat files.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        PSD_arr (np.ndarray): Array of the pore size distribution values for each diameter bin in d_arr.
    """

    d_arr = np.insert(np.arange(2*args.probe_radius, args.d_max + args.d_step, args.d_step),0,0)

    # Account for N_repeats
    if args.N_repeats > 1: PSD_arr = np.sum(PSD_arr.reshape(args.N_frames, args.N_repeats, -1), axis=1)

    PSD_arr = np.divide(PSD_arr.T, PSD_arr[:,0], dtype=float).T

    # Return the average and standard deviation (over the frames processed) of the probe-occupiable pore size distribution
    PSD_Cumulative = np.array([np.mean(PSD_arr, axis=0), np.std(PSD_arr, axis = 0)])
    # PSD is the negative derivative of the cumulative sum
    PSD = np.array([np.mean(-(PSD_arr[:,1:] - PSD_arr[:,:len(d_arr)-1])/(d_arr[1:] - d_arr[:len(d_arr)-1]), axis=0), np.std(-(PSD_arr[:,1:] - PSD_arr[:,:len(d_arr)-1])/(d_arr[1:] - d_arr[:len(d_arr)-1]), axis=0)])

    with open('Cumulative_PSD.dat', 'w') as anaout:
        print("# d (A) Cumulative_PSD Std", file=anaout)
        for i in range(len(PSD_Cumulative[0,:])):
            print(f" {np.round(d_arr[i], decimals=3):10.5f} {PSD_Cumulative[0,i]:10.5f} {PSD_Cumulative[1,i]:10.5f}", file=anaout)

    with open('PSD.dat', 'w') as anaout:
        print("# d (A) PSD Std", file=anaout) 
        for i in range(len(PSD_Cumulative[0,:])):
                if i == 0:
                    print(f" {np.round(d_arr[i], decimals=3):10.5f} {0.0:10.5f} {0.0:10.5f}", file=anaout)
                else:
                    print(f" {np.round(d_arr[i], decimals=3):10.5f} {PSD[0,i-1]:10.5f} {PSD[1,i-1]:10.5f}", file=anaout)


def write_FFV(args: argparse.Namespace, FFV_arr: np.ndarray) -> None:
    """Writes the FFV data to a .dat file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        FFV_arr (np.ndarray): Array of the free volume fraction values for each frame processed.
    """

    # Account for N_repeats
    if args.N_repeats > 1: FFV_arr = np.sum(FFV_arr.reshape(args.N_frames, args.N_repeats, -1), axis=1)
    FFV_c = FFV_arr[:,0] / FFV_arr[:,2]; FFV_lr = FFV_arr[:,1] / FFV_arr[:,2]

    # Return the average and standard deviation (over the frames processed) of the probe-occupiable fractional free volume
    FFV = np.array([np.mean(FFV_c), np.std(FFV_c),np.mean(FFV_lr), np.std(FFV_lr)])
    with open('FFV.dat', 'w') as anaout:
        print("# FFV Std - 0.0 = Connolly, 1.0 = Lee-Richards", file=anaout)
        print(f"0.0 {FFV[0]:10.5f} {FFV[1]:10.5f}", file=anaout)
        print(f"1.0 {FFV[2]:10.5f} {FFV[3]:10.5f}", file=anaout)


def write_Surface_area(Surface_area_arr: np.ndarray) -> None:
    """Writes the surface area data to a .dat file.
    
    Args:
        Surface_area_arr (np.ndarray): Array of the surface area values for each frame processed.
    """

    SA_c = Surface_area_arr[:,0]; SA_lr = Surface_area_arr[:,1]

    # Return the average and standard deviation (over the frames processed) of the surface area
    SA = np.array([np.mean(SA_c), np.std(SA_c), np.mean(SA_lr), np.std(SA_lr)])
    with open('SA.dat', 'w') as anaout:
        print("# SA (A^2) Std - 0.0 = Connolly, 1.0 = Lee-Richards", file=anaout)
        print(f"0.0 {SA[0]:15.5f} {SA[1]:10.5f}", file=anaout)
        print(f"1.0 {SA[2]:15.5f} {SA[3]:10.5f}", file=anaout)


def write_Tortuosity(tortuosity_arr: np.ndarray) -> None:
    """Writes the tortuosity data to a .dat file.
    
    Args:
        tortuosity_arr (np.ndarray): Array of the tortuosity values for each frame processed.
    """

    tortuosity_x = tortuosity_arr[:,0]; tortuosity_y = tortuosity_arr[:,1]; tortuosity_z = tortuosity_arr[:,2]

    # Return the average and standard deviation (over the frames processed) of the tortuosity
    if np.any(tortuosity_x == -1) or np.any(tortuosity_y == -1) or np.any(tortuosity_z == -1):
        tortuosity = np.array([-1, -1, -1, -1, -1, -1])
    else:
        tortuosity = np.array([np.mean(tortuosity_x), np.std(tortuosity_x), np.mean(tortuosity_y), np.std(tortuosity_y), np.mean(tortuosity_z), np.std(tortuosity_z)])

    with open('Tau.dat', 'w') as anaout:
        print("# Tortuosity Std - 0.0, 1.0, 2.0 = X, Y, and Z direction - value of -1 denotes a failed tortuosity analysis on at least 1 frame", file=anaout)
        print(f"0.0 {tortuosity[0]:10.5f} {tortuosity[1]:10.5f}", file=anaout)
        print(f"1.0 {tortuosity[2]:10.5f} {tortuosity[3]:10.5f}", file=anaout)
        print(f"2.0 {tortuosity[4]:10.5f} {tortuosity[5]:10.5f}", file=anaout)