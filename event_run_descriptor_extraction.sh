#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_run_descriptor_extraction.sh \"EVENT_NAME\" [ --force-start ] [ --overwrite-all ]
#
# --parallel N
#     -> N instances of descriptor extraction in parallel
#
# --force-start
#     -> videos already registered as being processed for another event are started anyway
#
# --clean-state-running
#     -> all videos marked as 'running' get a clean slate
#     -> no jobs will be launched with this option
#
# --overwrite-all
#     -> starts extraction for all videos, even those already processed
#
# RETURN VALUE (non-parallel): number of missing videos
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

set -u
source processing/usr/scripts/bash_utils.sh

##
## PARSE ARGUMENTS
##
EVENT_NAME=""
FORCE_START=NO
OVERWRITE_ALL=NO
CLEAN_STATE_RUNNING=NO
NB_INSTANCES=1

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
		--parallel)
		NB_INSTANCES="$2"
		shift
		re='^[0-9]+$'
		if ! [[ $NB_INSTANCES =~ $re ]] ; then
			echo "$USAGE_STRING"
			log_ERR "--parallel: Expecting an integer"
			exit 1
		fi
		;;
		--force-start)
		FORCE_START=YES
		;;
		--clean-state-running)
		CLEAN_STATE_RUNNING=YES
		;;
		--overwrite-all)
		OVERWRITE_ALL=YES
		;;
		*)
		# Event name here
		EVENT_NAME="$key"
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
log_INFO "Event ${EVENT_NAME}"

##########
# PARALLEL 
##########

if [ $NB_INSTANCES -gt 1 ]; then
	# This instance acts as the master.
	
	PIDS=()
	# Spawn instances
	for (( i = 0; i < ${NB_INSTANCES}; i++ )); do
		xterm -e "./event_run_descriptor_extraction.sh \"${EVENT_NAME}\"" &
		PIDS+=($!)
		log_OK "Spawned PID $!"
	done
	
	RETVAL=0
	# Wait for instances to finish
	trap 'echo Signal received - killing spawned processes.; kill -9 ${PIDS[@]}; exit 1' TERM HUP KILL INT
	for pid in ${PIDS[@]}; do
		log_INFO "Waiting for $pid.."
		wait $pid
		#RETVAL=$(( $RETVAL + $? ))
	done
	
	trap - TERM HUP KILL INT
	
	# Return value == add all instances' return values
	#if [ $RETVAL -eq 0 ]; then
	#	log_OK "Missing $RETVAL descriptors."
	#else
	#	log_ERR "Missing $RETVAL descriptors."
	#fi
	exit 0
fi


########################
# Function declaration #
########################

function cleanup {
	echo "Emergency exit on ${video}, cleaning up lock and status."
	rm -f "${LOCKFILE}"
	rm -f "${VIDEO_STATUS_FILE}"
}

function run_job_sequential {
	set -e
	# In case job is interrupted
	trap cleanup TERM HUP KILL INT
	
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
	
	trap - INT HUP TERM HUP
	
	set +e
	return 0
}

#### RUN JOBS

log_INFO "${TXT_BOLD}Checking all videos sequentially${TXT_RESET}"
log_TODO "Implement remote job handler, oar handler, etc."



# DENSETRACK
NB_DESCRIPTORS_MISSING=`cat "${COMP_DESC_QUEUE_FILE}" | wc -l`

while read -r video; do
	mkdir -p "${VIDS_WORK_DIR}${video}"
	LOCKFILE="${VIDS_WORK_DIR}${video}/descriptor_extraction.lock"
	VIDEO_STATUS_FILE="${VIDS_WORK_DIR}${video}/descriptor_extraction_status"
	VID_STATUS=`cat "${VIDEO_STATUS_FILE}" 2> /dev/null`
	
	if [[ ${CLEAN_STATE_RUNNING} == "YES" ]]; then
		if [[ ${VID_STATUS} == "running" ]]; then
			rm -f "${LOCKFILE}"
			rm -f "${VIDEO_STATUS_FILE}"
			log_OK "Cleared 'running' state for \"${video}\""
		fi
		continue
	fi
	
	# Quick checks to avoid locking unnecessarily
	
	
	if [[ "${VID_STATUS}" == "done" && ${OVERWRITE_ALL} == NO ]]; then
		log_OK "DenseTrack \"${video}\" already marked as done."
		NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
		continue
	elif [[ "${VID_STATUS}" == "running" && ${FORCE_START} == NO ]]; then
		log_WARN "DenseTrack job already registered for \"${video}\". Owner: \"`cat "$LOCKFILE"`\". See option \"--force-start\"."
		NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
		continue
	fi
	
	# START BY REGULAR MEANS, LOCKING
	if ( set -o noclobber; echo "$EVENT_NAME" > "$LOCKFILE") 2> /dev/null ; then
		# Lock acquired, re-check
		# The status can't be marked as "running" if lock was acquired
		VID_STATUS=`cat "${VIDEO_STATUS_FILE}" 2> /dev/null`
		if [[ "${VID_STATUS}" == "done" && ${OVERWRITE_ALL} == NO ]]; then
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
			NB_DESCRIPTORS_MISSING=$(( ${NB_DESCRIPTORS_MISSING} - 1 ))
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




