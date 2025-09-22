
# 📁 Folder Overview

This folder contains the scripts to run the proposed method in the partially data-driven case -the models are Universal Differential Equations (UDEs)- for the three dynamical systems: **damped oscillator**, **Lorenz system**, and **Lotka-Volterra** with $\mathcal{L}_{\text{acc}} = 10^{-3}$. The structure includes configuration files, result analyses, launcher scripts, and utility modules.

---

## 📂 Folder Structure

### 📊 Results Folders

- **`analysis_results_damped/`**
- **`analysis_results_lorenz/`**
- **`analysis_results_lv/`**  
  These folders contain the analysis of the results of the method proposed for each of the respective systems.

- **`results_maximized/`**  
  This folder contains the results as `.jld` files of the method proposed. For each dynamical system, the method was run starting from 10 different initializations, resulting in 10 corresponding result folders.

- **`training_UDE_results/`**  
  This folder contains the results of standard ensembling on the three different dynamical systems. One random model from each standard ensemble is used as the starting point for our method. The threshold to accept a model in the standard ensembles is $\mathcal{L}_{\text{acc}} = 10^{-3}$.

- **`identifiability_analysis_mechanistic_parameters/`**  
  This folder contains the results of the identifiability analysis (both with ensemble-based approach and with sensitivity-matrix approach) of the mechanistic parameters in the UDE models.

---

## ⚙️ Configuration and Utility Scripts

- `configurations_damped.jl`  
- `configurations_lorenz.jl`  
- `configurations_lv.jl`  
  Julia scripts containing parameter settings or simulation options for each system.

- `ConfidenceEllipse.jl`  
  Julia module containing functions to work with confidence ellipses of a multivariate normal distribution.

- `out_of_domain_variability.jl`  
- `out_of_domain_variability_3d.jl`  
  Julia modules for analyzing confidence ellipses of vector field predictions derived from an ensemble of models over a given region of the state space. The first script is customized for 2D vector fields, and the second for 3D vector fields.

- `data_generator/`  
  Scripts used for generating synthetic data for training.

---

## 🚀 Launcher Scripts

- `launcher_ensemble_damped.sh`  
- `launcher_ensemble_lorenz.sh`  
- `launcher_ensemble_lv.sh`  
  Bash scripts to replicate the paper results for each dynamical system.

---

## 📈 Method implementation

- `routing_loss_contour_damped.jl`  
- `routing_loss_contour_lorenz.jl`  
- `routing_loss_contour_lv.jl`  
  Main scripts that implement the method described in the paper, with $\mathcal{L}_{\text{acc}} = 10^{-3}$.

---

## 🧬 Project Metadata

- `Manifest.toml`  
- `Project.toml`  
  Julia environment files specifying project dependencies and versions.

---