#! /bin/sh
# Submit job to CVC cluster
# Jiaolong Xu, jiaolong@cvc.uab.es

# short.q : for processes that won’t last more than 2 hours using a maximum of 6GB RAM.
# short_big.q : for processes that won’t last more than 2 hours with no RAM limit.
# medium.q : for processes that won’t last more than 60 hours using a maximum of 6GB RAM.
# medium_big.q : for processes that won’t last more than 60 hours with no RAM limit.
# long.q : No time limit using a maximum of 6GB RAM.
# long_big.q : No time limit and no RAM limit.
# ise.q: reserved for the ISE group
# cic.q: reserved for the CIC group

# Example: qsub -q short.q -l mem=4G job.sh
qsub -q medium.q@compute-0-0 job.sh
