#!/bin/sh
#SBATCH --job-name=mod_ensemble_cell_apoptosis_model     
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giampiccolo@cosbi.eu     # Where to send email reports
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --cpus-per-task=5                  # Run on 20 tasks
#SBATCH --mem=50GB                          # Memory limit
#SBATCH --output=mod_ensemble_cell_apoptosis_model_1.out  # Standard output will be written to this file
#SBATCH --error=mod_ensemble_cell_apoptosis_model_1.err   # Standard error will be written to this file
#SBATCH --partition=cosbi                    # the partition to use, "cosbi" in our case
#SBATCH --account=cosbi                      # the account to use, "cosbi" in our case

######################
# Begin work section #
######################

echo "running mod generation cell apoptosis model"

/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 1
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 2
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 3
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 4
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 5
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 6
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 7
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 8
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 9
/cosbi/home/giampiccolo/.juliaup/bin/julia --threads 5 --project=../. routing_loss_contour_cell_ap.jl 10