#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_status.sh EVENT_NAME
#
# Reports status string and jobs currently in queue where applicable.
"

source processing/usr/scripts/bash_utils.sh

ERROR_COLOR="${TXT_BOLD}${TXT_RED}ERROR${TXT_RESET}"
WARNING_COLOR="${TXT_RED}WARNING${TXT_RESET}"

#
# PARSE ARGUMENT
#
if [[ "$1" == "" ]]; then
	echo "$USAGE_STRING"
	exit 1
fi
EVENT_NAME="$1"

set -e
set -u

echo      "--------------------"
#log_TITLE "${EVENT_NAME}"

#
# CHECK EXISTENCE
#
EV_DIR="events/${EVENT_NAME}"
WORK_DIR="$EV_DIR/workdir"
STATUS_FILE="$WORK_DIR/status"
POSITIVE_FILE="$WORK_DIR/_positive.txt"
BACKGROUND_FILE="$WORK_DIR/_background.txt"
COMP_DESC_QUEUE_FILE="$WORK_DIR/compute_descriptors_queue"
VIDS_WORK_DIR="processing/videos_workdir/"
CLASSIFIERS_DIR="$WORK_DIR/classifiers"

# EVENT NAME AND STATUS
#echo "${TXT_BOLD}Event: ${TXT_RESET}${EVENT_NAME}"
if [[ ! -e "$STATUS_FILE" ]]; then
	log_ERR "status file does not exist, state unclean"
	exit 1
fi

# VIDEOS WE HAVE
if [[ -e "$POSITIVE_FILE" && -e "$BACKGROUND_FILE" ]]; then
	NB_POS=`cat "${POSITIVE_FILE}" | wc -l`
	NB_BG=`cat "${BACKGROUND_FILE}" | wc -l`
	echo "$(( ${NB_POS} + ${NB_BG} )) training videos (${NB_POS} positive and ${NB_BG} background)."
else
	log_ERR "Couldn't find training video lists."
	exit 1
fi

NB_TOT=$(( $NB_POS + $NB_BG ))

# DESCRIPTORS PROCESSING
# -- running, done and waiting
NB_RUNNING=0
NB_DONE=0
NB_WAITING=0

set +e
while read -r video; do
	VID_STATUS=`cat "${VIDS_WORK_DIR}${video}/descriptor_extraction_status" 2> /dev/null`
	case "$VID_STATUS" in
		"running")
		NB_RUNNING=$(( $NB_RUNNING + 1 ))
		;;
		"done")
		NB_DONE=$(( $NB_DONE + 1 ))
		;;
		*)
		NB_WAITING=$(( $NB_WAITING + 1 ))
		;;
	esac
done < "$COMP_DESC_QUEUE_FILE"

if [[ $NB_DONE == $NB_TOT ]]; then
	echo -e "\n${TXT_BOLD}extraction: ${TXT_BG_GREEN}${TXT_WHITE}DenseTrack${TXT_RESET}"
elif [[ $NB_RUNNING > 0 ]]; then
	echo -e "\n${TXT_BOLD}extraction: ${TXT_BG_BLUE}${TXT_WHITE}DenseTrack${TXT_RESET} ${TXT_BOLD}(processing..)${TXT_RESET}"
else
	echo -e "\n${TXT_BOLD}extraction: ${TXT_BG_BLUE}${TXT_WHITE}DenseTrack${TXT_RESET}"
fi

echo "$NB_WAITING waiting."
echo "$NB_RUNNING running."
echo "$NB_DONE processed."

if [[ ! -e "$CLASSIFIERS_DIR" ]]; then
	exit 0
fi
echo -e "\n${TXT_BOLD}Stage 2: Classifier training${TXT_RESET}"
echo -e "Partial classifier available.\nTrained on 2015-03-26 with 56 positive and 35 background videos."

#if [[ "`cat $EVENT_NAME

#./processing/compute_descriptors.sh
#RES=$?

