#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_run_descriptor_extraction.sh
#
# Requires all components to be compiled.
"

source required_components/bash_utils.sh
set -e
set -u
ERROR_COLOR="${TXT_BOLD}${TXT_RED}ERROR${TXT_RESET}"
WARNING_COLOR="${TXT_RED}WARNING${TXT_RESET}"

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

EV_DIR="events/${EVENT_NAME}"
WORK_DIR="$EV_DIR/workdir"
mkdir -p "$WORK_DIR"

STATUS_FILE="$WORK_DIR/status"
if [[ "$POSITIVE_FILE" == "" ]]; then POSITIVE_FILE="$EV_DIR/positive.txt"; fi
if [[ "$BACKGROUND_FILE" == "" ]]; then BACKGROUND_FILE="$EV_DIR/background.txt"; fi
COMP_DESC_QUEUE_FILE="$WORK_DIR/compute_descriptors_queue"

echo -e "${TXT_BOLD}Creating \"$EVENT_NAME\"${TXT_RESET} (overwrite=${OVERWRITE}, ignore-missing=${IGNORE_MISSING})"
echo "Positive videos: \"$POSITIVE_FILE\""
echo "Background videos: \"$BACKGROUND_FILE\""
echo ""

#
# CHECK IF EVENT ALREADY EXISTS
#
if [[ -e "$STATUS_FILE" && $OVERWRITE == NO ]]; then
	echo "${ERROR_COLOR}: \"$EVENT_NAME\" is already initialized. Use --overwrite option to ignore this message."
	exit 1
fi

# CHECK BACKGROUND.TXT, POSITIVE.TXT
# FILL VIDEO LIST

VIDEOS=()

#
# 1) EXISTS
if [[ ! -e "$POSITIVE_FILE" || ! -e "$BACKGROUND_FILE" ]]; then
	echo "$USAGE_STRING"
	echo "${ERROR_COLOR}: expecting \"$POSITIVE_FILE\" and \"$BACKGROUND_FILE\""
	exit 1
fi

# 2) NON-EMPTY
if [[ `cat "$POSITIVE_FILE" | wc -l` < 1 || `cat "$BACKGROUND_FILE" | wc -l` < 1 ]]; then
	echo "$USAGE_STRING"
	echo "${ERROR_COLOR}: expecting non-empty \"$POSITIVE_FILE\" and \"$BACKGROUND_FILE\""
	exit 1
fi

# 3) VIDEOS EXIST
NB_MISSING=0
echo -n "" > missing_videos.txt
echo -n "" > "$WORK_DIR/_background.txt"
echo -n "" > "$WORK_DIR/_positive.txt"
sync

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
	echo -e "${WARNING_COLOR}: $NB_MISSING missing videos, see \"missing_videos.txt\" for the complete list."
	if [[ $IGNORE_MISSING == NO ]]; then
		echo "Use option --ignore-missing if you wish to continue anyway."
		exit 1
	fi
fi

#
# APPEND JOBS
#
echo -e "Found ${#VIDEOS[@]} videos.\n"

if [[ -e "$COMP_DESC_QUEUE_FILE" ]]; then
	echo "${WARNING_COLOR}: processing queue already exists, overwriting content."
fi
(
	for (( i=0; i < ${#VIDEOS[@]}; i++ )); do
		echo "${VIDEOS[${i}]}"
	done
) > "$COMP_DESC_QUEUE_FILE"
echo -e "Registered videos for processing in \"$COMP_DESC_QUEUE_FILE\".\n"


echo "${TXT_BOLD}Status:${TXT_RESET} ${TXT_GREEN}initialized${TXT_RESET}"
echo "initialized" > "$STATUS_FILE"


