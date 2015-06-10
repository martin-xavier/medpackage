#!/bin/bash
# INRIA LEAR team, 2015
# Xavier Martin xavier.martin@inria.fr
#
# Usage: run_json_score_server.sh [ --port INT ]
#
# Runs the json score server based on the scores present in "results/scores/".
# 
# Example requests with "--port 12080":
# curl http://localhost:12080/api_get_classifier_output?nb_results=10\&classifier_name=rock_climbing
