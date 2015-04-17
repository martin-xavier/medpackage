#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_init.sh \"EVENT_NAME\" [ --positive_vids POSITIVE_LIST.TXT ] [ --background_vids BG_LIST.TXT ] [ --overwrite ] [ --ignore-missing ]
#
#	--positive_vids POSITIVE_LIST.TXT
#		contains list of positive videos
#		default value: \"events/EVENT_NAME/positive.txt\"
#	--background_vids BG_LIST.TXT
#		list of neutral videos
#		default value: \"events/EVENT_NAME/background.txt\".
#
#		Info: the positive and background videos are considered as a single entity (no shot detection).
#
#	--overwrite
#		writes over existing event where applicable
#
#	--ignore-missing
#		continues even if some videos do not exist
#
# Prepares the directory structure, creates a job queue.
# Does not overwrite existing data unless explicitly asked.
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
BACKGROUND_FILE=""
POSITIVE_FILE=""
OVERWRITE=NO
IGNORE_MISSING=NO

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
		--background_vids)
		BACKGROUND_FILE="$2"
		shift
		;;
		--positive_vids)
		POSITIVE_FILE="$2"
		shift
		;;
		--overwrite)
		OVERWRITE=YES
		;;
		--ignore-missing)
		IGNORE_MISSING=YES
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
echo      "--------------------"
log_TITLE "Event initialization"

EV_DIR="events/${EVENT_NAME}"
WORK_DIR="$EV_DIR/workdir"
mkdir -p "$WORK_DIR"

STATUS_FILE="$WORK_DIR/status"
if [[ "$POSITIVE_FILE" == "" ]]; then POSITIVE_FILE="$EV_DIR/positive.txt"; fi
if [[ "$BACKGROUND_FILE" == "" ]]; then BACKGROUND_FILE="$EV_DIR/background.txt"; fi
COMP_DESC_QUEUE_FILE="$WORK_DIR/compute_descriptors_queue"

log_INFO "${TXT_BOLD}Creating \"$EVENT_NAME\"${TXT_RESET} (overwrite=${OVERWRITE}, ignore-missing=${IGNORE_MISSING})"
log_INFO "Positive videos: \"$POSITIVE_FILE\""
log_INFO "Background videos: \"$BACKGROUND_FILE\""
log_INFO ""

#
# CHECK IF EVENT ALREADY EXISTS
#
if [[ -e "$STATUS_FILE" && $OVERWRITE == NO ]]; then
	log_ERR "\"$EVENT_NAME\" is already initialized. Use --overwrite option to ignore this message."
	exit 1
fi

# CHECK BACKGROUND.TXT, POSITIVE.TXT
# FILL VIDEO LIST

VIDEOS=()

#
# 1) EXISTS
if [[ ! -e "$POSITIVE_FILE" || ! -e "$BACKGROUND_FILE" ]]; then
	echo "$USAGE_STRING"
	log_ERR "expecting \"$POSITIVE_FILE\" and \"$BACKGROUND_FILE\""
	exit 1
fi

# 2) NON-EMPTY
if [[ `cat "$POSITIVE_FILE" | wc -l` < 1 || `cat "$BACKGROUND_FILE" | wc -l` < 1 ]]; then
	echo "$USAGE_STRING"
	log_ERR "expecting non-empty \"$POSITIVE_FILE\" and \"$BACKGROUND_FILE\""
	exit 1
fi

# 3) VIDEOS EXIST
NB_MISSING=0
echo -n "" > missing_videos.txt
echo -n "" > "$WORK_DIR/_background.txt"
echo -n "" > "$WORK_DIR/_positive.txt"

while read -r video; do
	if [[ ! -e "videos/$video" ]]; then
		echo "$video" >> missing_videos.txt
		NB_MISSING=$(( $NB_MISSING + 1 ))
	else
		VIDEOS+=("$video")
		echo "$video" >> "$WORK_DIR/_positive.txt"
	fi
done < "$POSITIVE_FILE"
while read -r video; do
	if [[ ! -e "videos/$video" ]]; then
		echo "$video" >> missing_videos.txt
		NB_MISSING=$(( $NB_MISSING + 1 ))
	else
		VIDEOS+=("$video")
		echo "$video" >> "$WORK_DIR/_background.txt"
	fi
done < "$BACKGROUND_FILE"
if [[ $NB_MISSING > 0 ]]; then
	log_WARN "$NB_MISSING missing videos, see \"missing_videos.txt\" for the complete list."
	if [[ $IGNORE_MISSING == NO ]]; then
		log_INFO "Use option --ignore-missing if you wish to continue anyway."
		exit 1
	fi
fi

#
# APPEND JOBS
#
log_INFO "Found ${#VIDEOS[@]} videos."
log_INFO ""

if [[ -e "$COMP_DESC_QUEUE_FILE" ]]; then
	log_WARN "processing queue already exists, overwriting content."
fi
(
	for (( i=0; i < ${#VIDEOS[@]}; i++ )); do
		echo "${VIDEOS[${i}]}"
	done
) > "$COMP_DESC_QUEUE_FILE"
log_INFO "Registered videos for processing in \"$COMP_DESC_QUEUE_FILE\"."
log_INFO ""

log_TODO "Ask user for channels to use (default: DenseTrack)"

log_OK "Event \"${EVENT_NAME}\" initialized."
echo "initialized" > "$STATUS_FILE"

