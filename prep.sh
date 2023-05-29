#!/bin/bash

wget people.csail.mit.edu/hyluo/data/simple_data.tar.gz
tar -xzvf simple_data.tar.gz
rm simple_data.tar.gz

mkdir model_file
mkdir model_ft_file