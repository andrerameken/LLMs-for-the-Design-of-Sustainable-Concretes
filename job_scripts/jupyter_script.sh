#!/bin/env bash


ml purge


CONTAINER= # Path to container

apptainer exec --nv $CONTAINER jupyter notebook --config="${CONFIG_FILE}" 
