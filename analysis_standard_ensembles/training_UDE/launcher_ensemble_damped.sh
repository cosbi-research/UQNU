#!/bin/sh
#SBATCH --job-name=damped_ensemble_gain_3_1_0       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=BEGIN,END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giampiccolo@cosbi.eu     # Where to send email reports
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --cpus-per-task=20                  # Run on 20 tasks
#SBATCH --mem=200GB                          # Memory limit
#SBATCH --output=damped_ensemble_gain_3_1_0.out  # Standard output will be written to this file
#SBATCH --error=damped_ensemble_gain_3_1_0.err   # Standard error will be written to this file
#SBATCH --partition=cosbi                    # the partition to use, "cosbi" in our case
#SBATCH --account=cosbi                      # the account to use, "cosbi" in our case

######################
# Begin work section #
######################

echo "running damped_ensemble_gain_3_1_0"

/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 20 --project=../. ensemble_generator.jl 0.01 32 4 0 0.0 0.0 1.0 1 200 20 initialization_difference 3 5 0.1