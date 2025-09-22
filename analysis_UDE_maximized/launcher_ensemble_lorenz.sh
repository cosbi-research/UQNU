#!/bin/sh
#SBATCH --job-name=ensemble_constraint_lorenz_UDE       
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giampiccolo@cosbi.eu     # Where to send email reports
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --cpus-per-task=5                  # Run on 20 tasks
#SBATCH --mem=50GB                          # Memory limit
#SBATCH --output=ensemble_constraint_lorenz_1.out  # Standard output will be written to this file
#SBATCH --error=ensemble_constraint_lorenz_1.err   # Standard error will be written to this file
#SBATCH --partition=cosbi                    # the partition to use, "cosbi" in our case
#SBATCH --account=cosbi                      # the account to use, "cosbi" in our case

######################
# Begin work section #
######################

echo "running ensemble_constraint_lorenz"

/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 1
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 2
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 3
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 4
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 5
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 6
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 7
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 8
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 9
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=./.routing_loss_contour_lorenz.jl 10