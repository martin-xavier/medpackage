#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_run_training.sh \"EVENT_NAME\"
#
# VERBOSE mode available, set this flag:
#    export VERBOSE=1
"
#
# SWITCHING TO MED DIRECTORY
#
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
cd "${DIR}"

set -e
set -u
source processing/usr/scripts/bash_utils.sh

##
## PARSE ARGUMENTS
##
EVENT_NAME=""

while [[ $# > 0 ]]
	do
	key="$1"

	case $key in
#    	-e|--extension)
#	    EXTENSION="$2"
#	    shift
#	    ;;
		-h|--help)
		echo "$USAGE_STRING"
		exit 1
		;;
		*)
		# Event name here
		EVENT_NAME=$key
		;;
	esac
	shift
done

if [[ "$EVENT_NAME" == "" ]]; then
	echo "$USAGE_STRING"
	exit 1
fi

EV_DIR="events/${EVENT_NAME}"
WORK_DIR="$EV_DIR/workdir"
STATUS_FILE="$WORK_DIR/status"
COMP_DESC_QUEUE_FILE="$WORK_DIR/compute_descriptors_queue"
VIDS_WORK_DIR="processing/videos_workdir/"


echo      "--------------------"
log_TITLE "Classifier training"

source processing/config_MED.sh
python "processing/compute_classifiers/med_early_fusion.py" --train --event-class "${EVENT_NAME}" denseTrack_hog denseTrack_hof denseTrack_mbh

