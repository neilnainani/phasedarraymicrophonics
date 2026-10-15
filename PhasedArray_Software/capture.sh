#!/bin/bash


#Start with "chmod +x capture.sh" to make the script executable
#Then run the script with "./capture.sh <SAD_X_COORD> <SOD_Y_COORD>"
#Example: "./capture.sh 1.5 2.0"
#This will create a session code "Grid_X1.5_Y2.0" and save the audio to the INDIV directory
#The audio will be saved as "Mic1_Grid_X1.5_Y2.0.wav", "Mic2_Grid_X1.5_Y2.0.wav", etc.
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/INDIV" directory
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/SUM" directory
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/FIGS" directory
# Exit immediately if a command exits with a non-zero status
set -e

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: ./capture.sh <SAD_X_COORD> <SOD_Y_COORD> <Z_COORD> [DURATION_SEC] [COUNTDOWN_SEC]"
    echo "Example (defaults): ./capture.sh 1.5 2.0 1.2"
    echo "Example (custom): ./capture.sh 1.5 2.0 1.2 10 5"
    exit 1
fi

SAD_X=$1
SOD_Y=$2
Z_COORD=$3
DURATION=${4:-5}
COUNTDOWN=${5:-3}

# Map the coordinates to a session code format your dataset can read
SESSION_CODE="Grid_X${SAD_X}_Y${SOD_Y}_Z${Z_COORD}"
BASE_DIR="../MICRECORD/${SESSION_CODE}"
INDIV_DIR="${BASE_DIR}/INDIV"

# Create the standard Phased Array Microphonics directory structure
mkdir -p "$INDIV_DIR"
mkdir -p "${BASE_DIR}/SUM"
mkdir -p "${BASE_DIR}/FIGS"

echo "======================================"
echo "Session Code: $SESSION_CODE"
echo "Target Mic Position: X ($SAD_X), Y ($SOD_Y), Z ($Z_COORD)"
echo "Recording Duration: $DURATION seconds"
echo "Countdown Buffer: $COUNTDOWN seconds"
echo "Output Directory: $INDIV_DIR"
echo "======================================"

# The countdown and the RECORDING cue are printed by collect_dataset_point.py,
# so the cue appears only once the input stream is running (Python takes
# 1-2 s to start on Windows; a cue printed here would come too early).
echo "Loading audio libraries..."

# -------------------------------------------------------------------------
# HARDWARE CAPTURE
# This is where your wavCollection module takes over. 
# We call a dedicated python collection script that bypasses the 
# system.py beamforming and just records the raw data to the INDIV folder.
# -------------------------------------------------------------------------

python3 collect_dataset_point.py --code "$SESSION_CODE" --duration "$DURATION" --countdown "$COUNTDOWN"

echo "✅ Capture complete for X=$SAD_X, Y=$SOD_Y, Z=$Z_COORD!"
echo "Move target mic stand to the next grid point!"
echo "======================================"