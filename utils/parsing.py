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

# Helper functions for reading arguments

import argparse
import numpy as np
import os
import sys
from typing import Tuple, Dict, Any, Optional, List
import yaml

######################################################################
######################## Loading  Functions ##########################
######################################################################

def readable_file(path):
    """
    Check if a path exists and is a file.
    """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"The file '{path}' does not exist.")
    elif not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"The file '{path}' is not readable.")
    return path

def string2bool(input):
    """
    Convert string input to boolean
    """
    if input == 'True':
        return True
    elif input == 'False':
        return False
    
def string2none(input):
    """
    Convert string input to None
    """
    if input == 'None':
        return None

def int_range(min_val, max_val, min_incl, max_incl, negative_one):
    """
    Limit integer input to specified range
    """
    def int_range_checker(arg):
        try:
            f = int(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("Input is not an integer.")
        
        if negative_one and f == -1:
            return f
        
        if min_incl and max_incl and not (min_val <= f and f <= max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to integers in [{min_val}, {max_val}] or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to integers in [{min_val}, {max_val}]")
        elif (not min_incl) and max_incl and not (min_val < f and f <= max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to integers in ({min_val}, {max_val}] or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to integers in ({min_val}, {max_val}]")
        elif min_incl and (not max_incl) and not (min_val <= f and f < max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to integers in [{min_val}, {max_val}) or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to integers in [{min_val}, {max_val})")
        elif (not min_incl) and (not max_incl) and not (min_val < f and f < max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to integers in ({min_val}, {max_val}) or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to integers in ({min_val}, {max_val})")
        return f
        
    return int_range_checker

def float_range(min_val, max_val, min_incl, max_incl, negative_one):
    """
    Limit float input to specified range
    """
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("Input is not a float.")
        
        if negative_one and f == -1:
            return f
        
        if min_incl and max_incl and not (min_val <= f and f <= max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to floats in [{min_val}, {max_val}] or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to floats in [{min_val}, {max_val}]")
        elif (not min_incl) and max_incl and not (min_val < f and f <= max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to floats in ({min_val}, {max_val}] or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to floats in ({min_val}, {max_val}]")
        elif min_incl and (not max_incl) and not (min_val <= f and f < max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to floats in [{min_val}, {max_val}) or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to floats in [{min_val}, {max_val})")
        elif (not min_incl) and (not max_incl) and not (min_val < f and f < max_val):
            if negative_one:    raise argparse.ArgumentTypeError(f"Input restricted to floats in ({min_val}, {max_val}) or -1")
            else:               raise argparse.ArgumentTypeError(f"Input restricted to floats in ({min_val}, {max_val})")
        return f
        
    return float_range_checker


######################################################################
################# Main Function to Parse Arguments ###################
######################################################################

def load_Args() -> Tuple[argparse.Namespace, np.ndarray, np.ndarray, dict]:
    """
    Read in inputs from YAML file and command line using argparser.
        
    Returns:
        Tuple[argparse.Namespace, np.ndarray, np.ndarray, dict]: Parsed command-line arguments, array of atoms and vdW radii, array of dummy atom names, MDAnalysis Universe **kwargs input.
    """
    
    # Define parser for YAML config file
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('yaml_file', type = readable_file)

    # Add helpful error message if YAML file is not provided
    try:
        args, remaining_argv = config_parser.parse_known_args()
    except Exception as e:
        print(f'\nERROR parsing configuration: {e}')
        print('Please provide a valid YAML config file. Example: python3 PARSE.py config.yaml trj ...')
        sys.exit(1)

    # Load the YAML data
    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Define parser for inputs + help menu
    parser = argparse.ArgumentParser(description="PARSE: Pore Analysis & Reconstruction of Spatial Environments")
    # Define subparsers
    subparsers = parser.add_subparsers(dest="mode", help = "Input file mode", required=True)

    #################################
    #################################
    ## SUBPARSER 1: For .xyz files ##
    #################################
    #################################
    xyz_parser = subparsers.add_parser('xyz', help = "Process PoreBlazer-style xyz + dat trajectory files")
       ###################################
       ## GROUP 1: Required input files ##
       ###################################
    xyz_files = xyz_parser.add_argument_group('Required input files')
    xyz_files.add_argument('trj_file', type = readable_file,
                           help = "Path to xyz file")
    xyz_files.add_argument('top_file', type = readable_file,
                           help = "Path to dat file")
       ##########################################
       ## GROUP 2: Frame selection and threads ##
       ##########################################
    xyz_frames = xyz_parser.add_argument_group('Frame selection and threads')
    xyz_frames.add_argument('-n', '--N_frames', type = int, default = 1, choices = [1],
                            help = "Number of frames to analyze [Locked to 1 frame for xyz analysis]")
    xyz_frames.add_argument('--N_repeats', type = int_range(0.0, np.inf, False, False, False), default = config['N_repeats'],
                             help = "Number of times to analyze each frame. --N_repeats > 1 requires --Voxel_dist 'Random' [default = YAML]")
    xyz_frames.add_argument('-t', '--N_threads', type = int_range(0.0, np.inf, False, False, False), default = config['N_threads'],
                             help = "Number of threads for parallelization [default = YAML]")
       ###########################################
       ## GROUP 3: MDAnalysis selection strings ##
       ###########################################
    xyz_selection = xyz_parser.add_argument_group('MDAnalysis selection strings')
    xyz_selection.add_argument('-m', '--system_name', type = str, default = config['system_name'],
                               help = "MDAnalysis selection string defining the system matrix, e.g., 'all', 'moltype MOL', 'resname PEO', 'resname SOL LI CL' [default = YAML; Typically 'all' for PoreBlazer-style xyz + dat input.]")
    xyz_selection.add_argument('-s', '--solvent_name', type = str, default = config['solvent_name'],
                               help = "MDAnalysis selection string defining the solvent matrix, e.g., '', 'percolated', 'resname SOL LI CL' [default = YAML; Typically '' or 'percolated' for PoreBlazer-style xyz + dat input.]")
    xyz_selection.add_argument('--identify_atoms', type = str, choices = ['Names', 'Masses'], default = config['identify_atoms'],
                               help = "Method to identify the atom and associated vdW radii [default = YAML]")
       ##################################
       ## GROUP 4: Important variables ##
       ##################################
    vars = xyz_parser.add_argument_group('Important variables')
    vars.add_argument('-L', '--L_voxel', type = float_range(0.0, np.inf, False, False, False), default = config['L_voxel'],
                      help = "Voxel side length (A) [default = YAML]")
    vars.add_argument('-r', '--probe_radius', type = float_range(0.0, np.inf, False, False, False), default = config['probe_radius'],
                      help = "Probe radius (A) [default = YAML]")
    vars.add_argument('--d_max', type = float_range(0.0, np.inf, False, False, False), default = config['d_max'],
                      help = "Max PSD diameter (A) [default = YAML]")
    vars.add_argument('--d_step', type = float_range(0.0, np.inf, False, False, False), default = config['d_step'],
                      help = "PSD bin size (A) [default = YAML]")
    vars.add_argument('--Voxel_dist', type = str, choices = ['Uniform', 'Random'], default = config['Voxel_dist'],
                      help = "Voxel distribution setting [default = YAML; Locked to 'Uniform' or 'Random']")
    vars.add_argument('--PSD_FFV', type = string2bool, choices = [True, False], default = config['PSD_FFV'],
                      help = "Pore size distribution and free volume fraction calculation setting [default = YAML; Locked to True or False]")
    vars.add_argument('--Surface_area', type = string2bool, choices = [True, False], default = config['Surface_area'],
                      help = "Surface area calculation setting; Requires --Voxel_dist 'Uniform' and --tol -1 [default = YAML; Locked to True or False]")
    vars.add_argument('--Tortuosity', type = string2bool, choices = [True, False], default = config['Tortuosity'],
                      help = "Tortuosity calculation setting; Requires --Voxel_dist 'Uniform' and --tol -1 [default = YAML; Locked to True or False]")
       ###################################################
       ## GROUP 5: Terminal printing and xyz generation ##
       ###################################################
    printing = xyz_parser.add_argument_group('Terminal printing and xyz generation')
    printing.add_argument('--print_eff', type = int, choices = [0, 1, 2], default = config['print_eff'],
                          help = "Level of printing [default = YAML; Locked to 0, 1, or 2]")
    printing.add_argument('--print_xyz', type = string2bool, choices = [True, False], default = config['print_xyz'],
                          help = "xyz visualization flag [default = YAML; Locked to True or False]")
       ####################################
       ## GROUP 6: Efficiency parameters ##
       ####################################
    efficiency = xyz_parser.add_argument_group('Efficiency parameters - see YAML description for more details [default = YAML]')
    efficiency.add_argument('--clustering', type = str, choices = ['Neumann', 'Moore'], default = config['clustering'],)
    efficiency.add_argument('--N_calc_max', type = float_range(0.0, np.inf, False, False, False), default = config['N_calc_max'],)
    efficiency.add_argument('--N_write_max', type = float_range(0.0, np.inf, False, False, False), default = config['N_write_max'],)
    efficiency.add_argument('--d_inc', type = float_range(0.0, np.inf, False, False, False), default = config['d_inc'],)
    efficiency.add_argument('--N_edge_gen', type = float_range(0.0, np.inf, False, False, False), default = config['N_edge_gen'])
    efficiency.add_argument('--tol', type = float_range(0.0, 1.0, False, True, True), default = config['tol'],)
    efficiency.add_argument('--rand_frac', type = float_range(0.0, 1.0, False, True, False), default = config['rand_frac'],)

    ########################################################
    ########################################################
    ## SUBPARSER 2: For trajectory files (xtc, trr, etc.) ##
    ########################################################
    ########################################################
    traj_parser = subparsers.add_parser('trj', help = "Process GROMACS trajectory files")
       ###################################
       ## GROUP 1: Required input files ##
       ###################################
    trj_files = traj_parser.add_argument_group('Required input files')
    trj_files.add_argument('trj_file', type = readable_file,
                            help = "Path to xtc/trr/gro file")
    trj_files.add_argument('top_file', type = readable_file,
                            help = "Path to tpr/gro file")
       ##########################################
       ## GROUP 2: Frame selection and threads ##
       ##########################################
    traj_frames = traj_parser.add_argument_group('Frame selection and threads')
    traj_frames.add_argument('-b', '--t_min', type = float_range(0.0, np.inf, True, False, True), default = config['t_min'],
                             help = "Start time (ps) [default = YAML]")
    traj_frames.add_argument('-e', '--t_max', type = float_range(0.0, np.inf, True, False, True), default = config['t_max'],
                             help = "End time (ps) [default = YAML]")
    traj_frames.add_argument('-bi', '--start_idx', type = string2none or int, default = config['start_idx'],
                             help = "Start index [default = YAML]")
    traj_frames.add_argument('-ei', '--end_idx', type = string2none or int, default = config['end_idx'],
                             help = "End index [default = YAML]")
    traj_frames.add_argument('-n', '--N_frames', type = int_range(0.0, np.inf, False, False, True), default = config['N_frames'],
                             help = "Number of frames to analyze [default = YAML]")
    traj_frames.add_argument('--N_repeats', type = int_range(0.0, np.inf, False, False, False), default = config['N_repeats'],
                             help = "Number of times to analyze each frame. --N_repeats > 1 requires --Voxel_dist 'Random' [default = YAML]")
    traj_frames.add_argument('-t', '--N_threads', type = int_range(0.0, np.inf, False, False, False), default = config['N_threads'],
                             help = "Number of threads for parallelization [default = YAML]")
       ###########################################
       ## GROUP 3: MDAnalysis selection strings ##
       ###########################################
    traj_selection = traj_parser.add_argument_group('MDAnalysis selection strings')
    traj_selection.add_argument('-m', '--system_name', type = str, default = config['system_name'],
                                help = "MDAnalysis selection string defining the system matrix, e.g., 'moltype MOL', 'resname PEO', 'resname SOL LI CL' [default = YAML]")
    traj_selection.add_argument('-s', '--solvent_name', type = str, default = config['solvent_name'],
                                help = "MDAnalysis selection string defining the solvent matrix, e.g., '', 'percolated', 'resname SOL LI CL' [default = YAML]") 
    traj_selection.add_argument('--identify_atoms', type = str, choices = ['Names', 'Masses'], default = config['identify_atoms'],
                               help = "Method to identify the atom and associated vdW radii [default = YAML]")
       ##################################
       ## GROUP 4: Important variables ##
       ##################################
    vars = traj_parser.add_argument_group('Important variables')
    vars.add_argument('-L', '--L_voxel', type = float_range(0.0, np.inf, False, False, False), default = config['L_voxel'],
                      help = "Voxel side length (A) [default = YAML]")
    vars.add_argument('-r', '--probe_radius', type = float_range(0.0, np.inf, False, False, False), default = config['probe_radius'],
                      help = "Probe radius (A) [default = YAML]")
    vars.add_argument('--d_max', type = float_range(0.0, np.inf, False, False, False), default = config['d_max'],
                      help = "Max PSD diameter (A) [default = YAML]")
    vars.add_argument('--d_step', type = float_range(0.0, np.inf, False, False, False), default = config['d_step'],
                      help = "PSD bin size (A) [default = YAML]")
    vars.add_argument('--Voxel_dist', type = str, choices = ['Uniform', 'Random'], default = config['Voxel_dist'],
                      help = "Voxel distribution setting [default = YAML; Locked to 'Uniform' or 'Random']")
    vars.add_argument('--PSD_FFV', type = string2bool, choices = [True, False], default = config['PSD_FFV'],
                      help = "Pore size distribution and free volume fraction calculation setting [default = YAML; Locked to True or False]")
    vars.add_argument('--Surface_area', type = string2bool, choices = [True, False], default = config['Surface_area'],
                      help = "Surface area calculation setting; Requires --Voxel_dist 'Uniform' and --tol -1 [default = YAML; Locked to True or False]")
    vars.add_argument('--Tortuosity', type = string2bool, choices = [True, False], default = config['Tortuosity'],
                      help = "Tortuosity calculation setting; Requires --Voxel_dist 'Uniform' and --tol -1 [default = YAML; Locked to True or False]")
       ###################################################
       ## GROUP 5: Terminal printing and xyz generation ##
       ###################################################
    printing = traj_parser.add_argument_group('Terminal printing and xyz generation')
    printing.add_argument('--print_eff', type = int, choices = [0, 1, 2], default = config['print_eff'],
                          help = "Level of printing [default = YAML; Locked to 0, 1, or 2]")
    printing.add_argument('--print_xyz', type = string2bool, choices = [True, False], default = config['print_xyz'],
                          help = "xyz visualization flag [default = YAML; Locked to True or False]")
       ####################################
       ## GROUP 6: Efficiency parameters ##
       ####################################
    efficiency = traj_parser.add_argument_group('Efficiency parameters - see YAML description for more details [default = YAML]')
    efficiency.add_argument('--clustering', type = str, choices = ['Neumann', 'Moore'], default = config['clustering'],)
    efficiency.add_argument('--N_calc_max', type = float_range(0.0, np.inf, False, False, False), default = config['N_calc_max'],)
    efficiency.add_argument('--N_write_max', type = float_range(0.0, np.inf, False, False, False), default = config['N_write_max'],)
    efficiency.add_argument('--d_inc', type = float_range(0.0, np.inf, False, False, False), default = config['d_inc'],)
    efficiency.add_argument('--N_edge_gen', type = float_range(0.0, np.inf, False, False, False), default = config['N_edge_gen'])
    efficiency.add_argument('--tol', type = float_range(0.0, 1.0, False, True, True), default = config['tol'],)
    efficiency.add_argument('--rand_frac', type = float_range(0.0, 1.0, False, True, False), default = config['rand_frac'],)

    # Define args
    args = parser.parse_args(remaining_argv)

    if args.Voxel_dist == 'Uniform' and args.N_repeats != 1:        parser.error("--N_repeats 1 if --Voxel_dist 'Uniform'")
    # --Voxel_dist 'Uniform' and --tol -1 are required for SA calculations
    if args.Surface_area == True:
        if args.Voxel_dist != 'Uniform':                            parser.error("SA calculation requires --Voxel_dist 'Uniform'")
        if args.tol != -1:                                          parser.error("SA calculation requires --tol -1")
    # --Voxel_dist 'Uniform' and --tol -1 are required for Tau calculations
    if args.Tortuosity == True:
        if args.Voxel_dist != 'Uniform':                            parser.error("Tortuosity calculation requires --Voxel_dist 'Uniform'")
        if args.tol != -1:                                          parser.error("Tortuosity calculation requires --tol -1")
    # When calculating the PSD from all frames (--tol -1 or --rand_frac >= 0.5), --rand_frac 1 is the most efficient
    if args.tol == -1 or args.rand_frac >= 0.5: args.rand_frac = 1

    # Define data arrays from YAML
    Size_arr = np.array(config['Size_arr'], dtype=object)
    Dummy_atoms = np.array(config['Dummy_atoms'])
    mda_kwargs = config['MDAnalysis_Universe_kwargs']
    for key, value in mda_kwargs.items():
        if value == 'None': mda_kwargs[key] = None

    return args, Size_arr, Dummy_atoms, mda_kwargs