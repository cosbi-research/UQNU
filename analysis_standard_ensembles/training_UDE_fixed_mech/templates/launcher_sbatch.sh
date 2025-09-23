#!/bin/sh
#SBATCH --job-name=##MODEL##_##JOB_NAME##       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giampiccolo@cosbi.eu     # Where to send email reports
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --cpus-per-task=20                  # Run on 20 tasks
#SBATCH --mem=200GB                          # Memory limit
#SBATCH --output=##MODEL##_##JOB_NAME##.out  # Standard output will be written to this file
#SBATCH --error=##MODEL##_##JOB_NAME##.err   # Standard error will be written to this file
#SBATCH --partition=cosbi                    # the partition to use, "cosbi" in our case
#SBATCH --account=cosbi                      # the account to use, "cosbi" in our case

######################
# Begin work section #
######################

echo "running ##MODEL##_##JOB_NAME##"

/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 20 --project=../. ensemble_generator.jl ##SCRIPT_ARGUMENTS##