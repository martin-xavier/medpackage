#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin>at<inria.fr
USAGE_STRING="
# This script queries youtube for a given event name,
# downloads N first results and learns a classifier from them.
# Then, it computes scores over a given collection.
#
# Options:
#
#	--event-name \"EVENT_NAME\"
#
# 	--nb-positive NB
#		Number of positive videos to fetch from youtube.
#
#	--background-vids LIST_FILENAME
#		List of videos to use as background.
#
#   --score-vids LIST_FILENAME
#       List of videos to score.
#
#	-c|--channel CHANNEL
#		Add a descriptor channel.
#
#		
#	--parallel NB_PARALLEL
#		Instances of descriptor extraction.
#
#   --overwrite
#		Existing event with the same name will be overwritten.
#
#   --collection-dir DIR
#     If your events are defined in a separate collection directory, specify it here.
#     This defaults to the package's base directory, and dictates where descriptors will be saved.
#		
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
re='^[0-9]+$'

OVERWRITE=""
NB_INSTANCES=1
SHOT_SEPARATION=NO
CHANNELS=()
BG_VIDEOS=()
SCORE_VIDEOS=()

NB_INSTANCES=1
NB_POSITIVE=1

# hidden option
EVENT_NAME=""

COLLECTION_DIR=${COLLECTION_DIR:="./"}

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
		--nb-positive)
		NB_POSITIVE="$2"
		shift
		if ! [[ $NB_POSITIVE =~ $re ]] ; then
			echo "$USAGE_STRING"
			log_ERR "--nb-positive: Expecting an integer"
			exit 1
		fi
		;;
		--background-vids)
		while read -r video; do
			BG_VIDEOS+=( "$video" )
		done < "$2"
		shift
		;;
		--score-vids)
		while read -r video; do
			SCORE_VIDEOS+=( "$video" )
		done < "$2"
		shift
		;;
		-c|--channel)
		log_INFO "Using channel $2"
		CHANNELS+=( "$2" )
		shift
		;;
		--parallel)
		NB_INSTANCES="$2"
		if ! [[ $NB_INSTANCES =~ $re ]] ; then
			echo "$USAGE_STRING"
			log_ERR "--parallel: Expecting an integer"
			exit 1
		fi
		shift
		;;
		--overwrite)
		OVERWRITE="--overwrite"
		;;
		--event-name)
		# hidden option
		EVENT_NAME="$2"
		shift
		;;
		--collection-dir)
		COLLECTION_DIR="$2"
		shift
		;;
		*)
		# Event name here
		#EVENT_NAME="$key"
		;;
	esac
	shift
done


if [ "$EVENT_NAME" == "" ]; then
	echo "${USAGE_STRING}"
	log_ERR "Need to provide a youtube query string (see --event-name)."
	exit 1
elif [[ ${#BG_VIDEOS[@]} -lt 1 ]]; then
	echo "${USAGE_STRING}"
	log_ERR "Need more background videos, see --background-vids"
	exit 1
elif [[ ${#SCORE_VIDEOS[@]} -lt 1 ]]; then
	echo "${USAGE_STRING}"
	log_ERR "Need more videos to score, see --score-vids"
	exit 1
fi


log_TITLE "AXES-Lite On-the-fly class creator"

set -e

NB_PROC=`cat /proc/cpuinfo | grep processor | wc -l`
re='^[0-9]+$'
if ! [[ $NB_POSITIVE =~ $re ]] ; then
	log_WARN "Couldn't parse '/proc/cpuinfo', launching 4 threads."
	NB_PROC=4
fi

UUID=`uuidgen`
TMPFILE="/tmp/med_onthefly_${UUID}.tmp"

# Create the event
echo "Creating event.."

( for (( i = 0; i < ${#BG_VIDEOS[@]}; i++ )); do echo ${BG_VIDEOS[${i}]}; done ) > ${TMPFILE}
./youtube_create_event.sh --event-name "${EVENT_NAME}" --parallel ${NB_PROC} --background-vids ${TMPFILE} -c denseTrack ${OVERWRITE} --nb-positive ${NB_POSITIVE}

# Run scores
echo "Scoring videos.."

( for (( i = 0; i < ${#SCORE_VIDEOS[@]}; i++ )); do echo ${SCORE_VIDEOS[${i}]}; done ) > ${TMPFILE}
./video_run_scores.sh --event "${EVENT_NAME}" --video-list ${TMPFILE} --output-directory "${COLLECTION_DIR}/scores_MED/"

# Signal LIMAS
echo "Signaling LIMAS.."
#ssh bbc_demo /home/axes/servers/wp6/limas-${DATASET}/updateEvents.sh

echo "DONE"

