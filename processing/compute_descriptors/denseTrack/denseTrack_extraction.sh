#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr
USAGE_STRING="
# Usage: denseTrack_extraction.sh VIDNAME --scenecutfile FILE
#
# The path of the video is assumed to start from the MED_BASEDIR/../videos/ directory.
#
# Requires all components to be compiled.
"

source "${MED_BASEDIR}usr/scripts/bash_utils.sh"

SCENECUTFILE=""
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
		exit 0
		;;
		--scenecutfile)
		SCENECUTFILE="$2"
		if [ ! -e "${SCENECUTFILE}" ]; then
			log_ERR "Scenecut file \"${SCENECUTFILE}\" does not exist."
			exit 1
		fi
		shift
		;;
		*)
		# Video name here
		VIDNAME="$key"
		;;
	esac
	shift
done

# Check that video exists
if [ ! -e "${MED_BASEDIR}/../videos/${VIDNAME}" ]; then
	log_ERR "Video \"${MED_BASEDIR}/../videos/${VIDNAME}\" does not exist"
	exit 1
fi

# Check that DenseTrackStab exists
which DenseTrackStab 1>/dev/null 2>/dev/null
if [ $? -ne 0 ]; then
	log_ERR "Couldn't find DenseTrackStab in PATH. Compile the dense trajectories executable and make sure it is available in your PATH environment variable."
	exit 1
fi

if [ "${SCENECUTFILE}" == "" ]; then
	python "${MED_BASEDIR}/compute_descriptors/denseTrack/densetrack_to_fisher_shot_errorprotect.py" --video "${VIDNAME}" --split train -k 256 --redo --slice 1 --save slice --featurepath "${MED_BASEDIR}videos_workdir/${VIDNAME}"
else
	python "${MED_BASEDIR}/compute_descriptors/denseTrack/densetrack_to_fisher_shot_errorprotect.py" --video "${VIDNAME}" --split train -k 256 --redo --slice 1 --save slice --scenecut "${SCENECUTFILE}" --featurepath "${MED_BASEDIR}videos_workdir/${VIDNAME}/shots"
fi

# CHECK RESULTS
set -e
for i in `cat "${MED_BASEDIR}compute_descriptors/denseTrack/denseTrack_descriptors.list"`;
do
	if [[ "${SCENECUTFILE}" == "" && ! -e "${MED_BASEDIR}/videos_workdir/${VIDNAME}/${i}.fvecs" ]]; then
		echo "1: looking for ${MED_BASEDIR}/videos_workdir/${VIDNAME}/${i}.fvecs"
		log_ERR "Descriptors not generated for \"${VIDNAME}\"."
		exit 1
	elif [[ ! "${SCENECUTFILE}" == "" && ! -e "${MED_BASEDIR}/videos_workdir/${VIDNAME}/shots/${i}.fvecs" ]]; then
		echo "2: looking for ${MED_BASEDIR}/videos_workdir/${VIDNAME}/shots/${i}.fvecs"
		log_ERR "Descriptors not generated for \"${VIDNAME}\"."
		exit 1
	fi
done

exit 0

