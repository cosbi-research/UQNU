
# 📁 Folder Overview

This folder contains the scripts to analyze the results of partial reconstruction (with fixed mechanistic parameters) for the **Lorenz** system.

---

## ⚙️ Analysis Scripts

- `evaluation_ensembles_on_vector_fields.jl` julia script to analyze the coverage probability of MOD ensembles on the vector field 
- `evaluation_standard_ensembles_on_vector_fields.jl` julia script to analyze the coverage probability of standard ensembles on the vector field 
- `evaluation_ensemble_on_vector_fields_projected.jl` julia script to analyze the coverage probability of MOD ensembles on the vector field projected to 2D regression plane
- `evaluation_standard_ensemble_on_vector_fields_projected.jl` julia script to analyze the coverage probability of standard ensembles on the vector field projected to 2D regression plane 
- `comparison_area.jl` julia script to compare the areas of confidences region on the vector field generated with standard and MOD ensembles
- `evaluation_ensembles_on_vector_fields_extended.jl`  julia script to analyze the coverage probability of MOD ensembles on the vector field (extended region)
- `evaluation_standard_ensembles_on_vector_fields_extended.jl` julia script to analyze the coverage probability of standard ensembles on the vector field (extended region)
- `evaluation_ensemble_on_vector_fields_projected_extended.jl`  julia script to analyze the coverage probability of MOD ensembles on the vector field (extended region) projected to 2D regression plane
- `evaluation_standard_ensemble_on_vector_fields_projected_extended.jl` julia script to analyze the coverage probability of standard ensembles on the vector field (extended region) projected to 2D regression plane
- `evaluation_uncertainty_on_trajectories.jl` julia script to analyze the coverage probability of MOD ensembles on the test trajectories
- `evaluation_standard_uncertainty_on_trajectories.jl` julia script to analyze the coverage probability of standard ensembles on the test trajectories
- `anlysis_results_on_trajectories.jl` julia script to compare the coverage probabilities between MOD ensembles and standard ensembles on the test trajectries
- `comparison_cp.jl` julia script to compare the coverage probabilities of confidence regions on vector field and test trajectories generated with standard and MOD ensembles
- `evaluation_on_training_trajectories.jl` julia script to analyze the accuracy of MOD ensembles on the training trajectories
- `evaluation_standard_on_training_trajectories.jl` julia script to analyze the accuracy of standard ensembles on the training trajectories
- `evaluation_cost_on_training_trajectories.jl` julia script to compare the accuracy on training trajectories between standard and MOD ensembles
- `analysis_qualitative_features_vector_fields.jl` julia script to evaluate if MOD ensembles preserve qualitative features of the ground-truth vector field
- `evaluation_stopping_criterion.jl` julai script to evaluate the behaviour of the objective function (quantifying the disagreeement) 
---

## 🚀 Launcher Scripts

- `analysis_launcher.bat`  
  Bash scripts to run all the analysis scripts

---