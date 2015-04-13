#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr

USAGE_STRING="
# Usage: EXAMPLE_run.sh
"

source required_components/bash_utils.sh
set -e
set -u
ERROR_COLOR="${TXT_BOLD}${TXT_RED}ERROR${TXT_RESET}"
WARNING_COLOR="${TXT_RED}WARNING${TXT_RESET}"



log_OK "Running test suite"
log_ERR "Error: this isn't patrick"
log_WAR "Warning: this is visible"
