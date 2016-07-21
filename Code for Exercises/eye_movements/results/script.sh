#!/bin/bash


# for obs in "ipr" "kvdb" "sk" "to" "tp" "vf"; do
# for obs in "ipa"; do
#     for sess in `seq 1 5`; do
# #         mkdir ${obs}
#        scp marianne@deighton.bccn-berlin.de:/home/mkp/search_eyemov_ctd/results/${obs}/${obs}_${sess}.edf ${obs}
#        edf2asc -neye -s ${obs}/${obs}_${sess}.edf
#        python extract_time_position.py ${obs} ${sess}
#        rm ${obs}/${obs}_${sess}.edf ${obs}/${obs}_${sess}.asc
#     done
# done


for obs in "kvdb" "sk" "to" "tp" "vf"; do
    for fix_alg in "dispersion" "velocity"; do
        python detect_fixations.py ${obs} ${fix_alg}
    done
done