#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: event_status.sh EVENT_NAME
#
# Reports status string and jobs currently in queue where applicable.
"

source required_components/bash_utils.sh

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

#
# CHECK EXISTENCE
#
EV_DIR="events/${EVENT_NAME}"
WORK_DIR="$EV_DIR/workdir"
STATUS_FILE="$WORK_DIR/status"
POSITIVE_FILE="$WORK_DIR/_positive.txt"
BACKGROUND_FILE="$WORK_DIR/_background.txt"
COMP_DESC_QUEUE_FILE="$WORK_DIR/compute_descriptors_queue"


# Event name
	echo "${TXT_BOLD}Event: ${TXT_RESET}${EVENT_NAME}"
# Status
	if [[ -e "$STATUS_FILE" ]]; then
		echo "${TXT_BOLD}Status:${TXT_RESET} `cat "$STATUS_FILE"`"
	else
		log_ERR "Status: does not exist or state unclean"
#		echo "${TXT_BOLD}Status:${TXT_RESET} ${TXT_RED}${TXT_BOLD}does not exist or state unclean${TXT_RESET}"
#		exit 0
	fi

echo ""
# Video status
	if [[ -e "$POSITIVE_FILE" && -e "$BACKGROUND_FILE" ]]; then
		NB_POS=`cat "${POSITIVE_FILE}" | wc -l`
		NB_BG=`cat "${BACKGROUND_FILE}" | wc -l`
		echo "$(( ${NB_POS} + ${NB_BG} )) training videos (${NB_POS} positive and ${NB_BG} background)."
	else
		echo "${ERROR_COLOR}: Couldn't find training video lists."
	fi
# State of queue
	if [[ -e "$COMP_DESC_QUEUE_FILE" ]]; then
		echo "`cat "${COMP_DESC_QUEUE_FILE}" | wc -l` queued for processing."
	else
		echo "${ERROR_COLOR}: Couldn't find processing queue."
	fi
	
	echo "0 currently being processed."
	echo "0 processed."

#if [[ "`cat $EVENT_NAME

#./processing/compute_descriptors.sh
#RES=$?

