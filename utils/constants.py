# Contains some constants used across multiple files in the PARSE package. This file is imported by other files to ensure consistency in the use of these constants.

import numpy as np

# Set the float data type used for atom coordinates and free volume sphere radii
# Always try np.float64 first
# If memory is an issue, use np.float32 - may introduce some error due to lack of precision
FLOAT_TYPE = np.float64