#!/bin/bash

if [[ ! -e ../config.json || ! -e ../config.json.template ]]; then
	echo "Couldn't find '../config.json' or '../config.json.template'."
	exit 1
fi

echo "Creating soft links to 'config.json' and 'config.json.template'."
ln -s ../config.json config.json
ln -s ../config.json.template config.json.template
echo "Done."
