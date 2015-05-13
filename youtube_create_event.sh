# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr
USAGE_STRING="
# This script queries youtube for a given event name,
# downloads N first results and learns a classifier from them.
"

set -u
source processing/usr/scripts/bash_utils.sh

log_TITLE "Automated class creator"

##
## PARSE ARGUMENTS
##
re='^[0-9]+$'

FORCE_START=NO
OVERWRITE=""
CLEAN_STATE_RUNNING=NO
NB_INSTANCES=1
SHOT_SEPARATION=NO
CHANNELS=()
BG_VIDEOS=()

NB_INSTANCES=1

# hidden option
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
		--background-vids)
		while read -r video; do
			BG_VIDEOS+=( "$video" )
		done < "$2"
		shift
		;;
		-c|--channel)
		log_INFO "Using channel $2"
		CHANNELS+=( "$2" )
		shift
		;;
		--shot-separation)
		if [[ "$2" == "NO" || "$2" == "YES" ]]; then
			SHOT_SEPARATION=$2
		else
			echo "$USAGE_STRING"
			log_ERR "Invalid value for --shot-separation (expecting YES or NO)."
		fi
		shift
		;;
		--parallel)
		NB_INSTANCES="$2"
		shift
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
		--overwrite)
		OVERWRITE="--overwrite"
		;;
		--event-name)
		# hidden option
		EVENT_NAME="$2"
		shift
		;;
		*)
		# Event name here
		EVENT_NAME="$key"
		;;
	esac
	shift
done

if [ "${EVENT_NAME}" == "" ]; then
	log_ERR "Need to specify an event name"
	exit 1
fi

# NEED AT LEAST ONE CHANNEL
if [[ ${#CHANNELS[@]} -eq 0 ]]; then
	echo "$USAGE_STRING"
	log_ERR "Need at least one channel, see option -c|--channel."
	exit 1
fi

if [[ ${#BG_VIDEOS[@]} -eq 0 ]]; then
	echo "$USAGE_STRING"
	log_ERR "Need at least one background video, see option --background-vids."
	exit 1
fi

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






# CHECK ELINKS IS AVAILABLE
ELINKS=$( which elinks )
if [ "${ELINKS}" == "" ]; then
	log_ERR "Package \"elinks\" is required to fetch the youtube result list."
	exit 1
fi

# CHECK YOUTUBE-DL IS AVAILABLE
YOUTUBE_DL=$( which youtube-dl )
if [ "${YOUTUBE_DL}" == "" ]; then
	log_ERR "Package \"youtube-dl\" is required to download the videos."
	exit 1
fi

set -e

log_INFO "Fetching list of youtube hits..."
elinks -dump 1 -no-numbering "https://www.youtube.com/results?lclk=short&search_query=${EVENT_NAME}&filters=short" > elinks_out.tmp
VIDEOS=( $( cat elinks_out.tmp | grep watch | awk -F "watch\\\\?v=" '{ print $2 }' | uniq ) )
rm elinks_out.tmp

log_OK "Found ${#VIDEOS[@]} videos."


for (( i = 0; i < ${#VIDEOS[@]}; i++ )); do
	log_IMPORTANT "Downloading video $(( ${i} + 1 ))/${#VIDEOS[@]}."
	youtube-dl -f best -o "videos/_youtube/${VIDEOS[${i}]}" ${VIDEOS[${i}]};
done

log_OK "Videos downloaded. Initializing event..."

EVENT_DIR="events/${EVENT_NAME}"
mkdir -p "${EVENT_DIR}"
( for (( i = 0; i < ${#VIDEOS[@]}; i++ )); do
	echo "_youtube/${VIDEOS[${i}]}"
done ) > "${EVENT_DIR}/positive.txt"

( for (( i = 0; i < ${#BG_VIDEOS[@]}; i++ )); do
	echo "${BG_VIDEOS[${i}]}"
done ) > "${EVENT_DIR}/background.txt"

./event_init.sh "${EVENT_NAME}" ${OVERWRITE}

./event_run_descriptor_extraction.sh "${EVENT_NAME}" --parallel ${NB_INSTANCES}

./event_run_training.sh "${EVENT_NAME}"

