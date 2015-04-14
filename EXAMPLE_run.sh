#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: EXAMPLE_run.sh
"

source processing/usr/scripts/bash_utils.sh
set -e
set -u
ERROR_COLOR="${TXT_BOLD}${TXT_RED}ERROR${TXT_RESET}"
WARNING_COLOR="${TXT_RED}WARNING${TXT_RESET}"
trap 'test_ERR "stage failed"; exit 1' ERR

EVENT_NAME="biking UCF-101"

#test_INFO "Creating test event"
#test_OK "Running test suite"
#test_ERR "Error: not working"
#test_WARN "Warning: this is visible"

rm -rf "events/${EVENT_NAME}/workdir/"
./event_init.sh "$EVENT_NAME"

#test_INFO "Checking status"

./event_status.sh "$EVENT_NAME"

#test_INFO "Computing descriptors"

./event_run_descriptor_extraction.sh "$EVENT_NAME" --force-start

#test_INFO "Check on progress.."

./event_status.sh "$EVENT_NAME"

