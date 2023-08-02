#!/bin/bash

while getopts "t:" option; do
    case "${option}" in
        t)
            type="$OPTARG";;
    esac
done

echo $type
echo "Experiment type = $type"

if [[ $type = "food" ]]
then 
    echo "Running food experiment"

elif [[ $type = "parameter_space" ]]
then
    echo "Running parameter_space experiment"

else
    echo "Unrecognized experiment type"
fi













# $OPTIND=1

#!/bin/bash

# usage() { echo "Usage: $0 [-s <45|90>] [-p <string>]" 1>&2; exit 1; }

# while getopts ":s:p:" o; do
#     case "${o}" in
#         s)
#             s=${OPTARG}
#             ((s == 45 || s == 90)) || usage
#             ;;
#         p)
#             p=${OPTARG}
#             ;;
#         *)
#             usage
#             ;;
#     esac
# done
# shift $((OPTIND-1))

# if [ -z "${s}" ] || [ -z "${p}" ]; then
#     usage
# fi

# echo "s = ${s}"
# echo "p = ${p}"
