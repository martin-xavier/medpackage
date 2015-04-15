#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_run_descriptor_extraction.sh EVENT_NAME [ --force-start ] [ --overwrite-all ]
#
# --force-start
#     -> videos already registered as being processed for another event are started anyway
#
# --overwrite-all
#     -> starts extraction for all videos, even those already processed
#
# RETURN VALUE:
#   number of missing videos (0 == all videos are processed)
#
# Requires all components to be compiled.
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
FORCE_START=NO
OVERWRITE_ALL=NO

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
		--force-start)
		FORCE_START=YES
		;;
		--overwrite-all)
		OVERWRITE_ALL=YES
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
log_TITLE "Descriptor extraction"


########################
# Function declaration #
########################

function cleanup {
	echo "Emergency exit on ${video}, cleaning up lock and status."
	rm -f "${LOCKFILE}"
	rm -f "${VIDEO_STATUS_FILE}"
}

function run_job_sequential {
	# In case job is interrupted, 
	trap cleanup INT TERM EXIT
	
	echo "running" > "$VIDEO_STATUS_FILE"
	
	log_INFO "Launching job for \"$video\"."
	
	(
	source processing/config_MED.sh
	densetrack_and_fisher_fullvid.sh "${video}"
	exit $?
	)
	
	if [[ $? == 0 ]]; then
	## "done" message must be written when you've checked the results are those expected
		echo "done" > "$VIDEO_STATUS_FILE"
		log_OK "Job successful for \"$video\"."
	else
		rm -f "$VIDEO_STATUS_FILE"
		log_ERROR "Job failed for \"${video}\"."
	fi
	
	rm -f "${LOCKFILE}"
	
	trap - INT TERM EXIT
	
	return 0
}

#### RUN JOBS

log_INFO "${TXT_BOLD}Computing descriptors sequentially on the whole dataset${TXT_RESET}"
log_TODO "Implement parallel jobs, oar handler, etc."



# DENSETRACK
NB_DESCRIPTORS_MISSING=`cat "${COMP_DESC_QUEUE_FILE}" | wc -l`

set -e
while read -r video; do
	mkdir -p "${VIDS_WORK_DIR}${video}"
	LOCKFILE="${VIDS_WORK_DIR}${video}/descriptor_extraction.lock"
	VIDEO_STATUS_FILE="${VIDS_WORK_DIR}${video}/descriptor_extraction_status"
	
	# Quick check to avoid locking unnecessarily
	if [[ `cat "${VIDEO_STATUS_FILE}" 2> /dev/null` == "done" && ${OVERWRITE_ALL} == NO ]]; then
		log_OK "DenseTrack \"${video}\" already marked as done."
		NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
		continue
	fi
	
	# START BY REGULAR MEANS, LOCKING
	if ( set -o noclobber; echo "$EVENT_NAME" > "$LOCKFILE") 2> /dev/null ; then
		# Lock acquired, re-check
		if [[ `cat "${VIDEO_STATUS_FILE}" 2> /dev/null` == "done" && ${OVERWRITE_ALL} == NO ]]; then
			log_OK "DenseTrack \"${video}\" already marked as done."
			NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
			rm -f "${LOCKFILE}"
			continue
		fi
		
		# Launch job
		run_job_sequential
		# {Lock released}
		
		if [[ $? == 0 ]]; then
			NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
		fi
	else
		if [[ ${FORCE_START} == NO ]]; then
			log_WARN "DenseTrack job already registered for \"${video}\". Owner: \"`cat "$LOCKFILE"`\". See option \"--force-start\"."
		else
			# Force start
			log_WARN "Force start on DenseTrack \"${video}\". Original owner: \"`cat "$LOCKFILE"`\""
			run_job_sequential
			if [[ $? == 0 ]]; then
				NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
			fi
		fi
	fi
done < "${COMP_DESC_QUEUE_FILE}"
set -e

log_INFO "Missing ${NB_DESCRIPTORS_MISSING} descriptors."
exit ${NB_DESCRIPTORS_MISSING}




