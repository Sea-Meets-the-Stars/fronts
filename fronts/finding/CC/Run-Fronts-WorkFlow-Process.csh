#!/bin/bash

# This script runs programs in the fronts/gradient workflow. In order to get the right program for
# this computer, get the computer name first. The computer name is the HOSTNAME minus the network
# location, either .local or .gso.uri.edu.

tempname=${HOSTNAME%".local"}
name_of_this_computer=${tempname%".gso.uri.edu"}
name_of_this_computer=${name_of_this_computer%".po"}
echo $name_of_this_computer

# Run AppQualMed 

#./AppQualMed_CMS_Main-$name_of_this_computer <<EOF
./$1-$name_of_this_computer <<EOF
`echo $2`
`echo $3`
`echo $4`
`echo $5`
`echo $6`
#EOF

