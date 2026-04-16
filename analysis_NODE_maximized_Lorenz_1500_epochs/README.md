
# 📁 Folder Overview

This folder contains the scripts to run the proposed method in the purely data-driven case -the models are pure Neural Ordinary Differential Equations (NODEs)- for the dynamical system **Lorenz system**, comparing the results obtained with 800 and 1500 training epochs. The structure includes configuration files, result analyses, launcher scripts, and utility modules.

---

## 📂 Folder Structure

### 📊 Results Folders

- **`analysis_results_lorenz/`**
  These folders contain the analysis of the results of the method proposed.

- **`results_maximized_800_epochs/`**  
  This folder contains the results as `.jld` files of the method proposed run for 800 epochs.

- **`results_maximized_1500_epochs/`**  
  This folder contains the results as `.jld` files of the method proposed run for 1500 epochs.

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

- `launcher_ensemble_lorenz.sh`  
  Bash scripts to replicate the results with 1500 epochs (the result with 800 epochs are computed in "analysis_NODE_maximized" folder).

---

## 📈 Method implementation

- `routing_loss_contour_lorenz.jl`  
  Main scripts that implement the method with 1500 epochs.

---

## 🧬 Project Metadata

- `Manifest.toml`  
- `Project.toml`  
  Julia environment files specifying project dependencies and versions.

---