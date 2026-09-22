# BOLT-release

Peptide-domain branch: this tree carries only the antimicrobial peptide design
experiments. The LLVM / query-plan domain lives on its own branch.

## Repository Structure

### `fine-tuning/`
Scripts and data to fine-tune Large Language Models (LLMs) for generating better
initializations.
- **`peptides/`**: Training-data generation and sampling for peptide design, plus
  `torchtune_config/` with the YAML configs used for SFT and DPO.

### `optimization/`
The core BO algorithms.
- **`peptides/`**: Implements the BO loop for peptide design.
  - **`lolbo/`**: Core logic for Latent Space BO.
  - **`apex_oracle/`**: Oracle for evaluating peptide properties.
  - **`uniref_vae/`**: VAE models for peptide sequences.

### `peptide_experiment/`
Training orchestration: the trajectory chain that interleaves BO sampling with
SFT/DPO fine-tuning, plus the MTBO / OptFormer / GP-expert baseline trainers.
Training only — it contains no evaluation code. See its README.

### `experiments/eval2/`
Everything evaluation: the compute engines, the train/eval pipeline runbooks, and
the `paper/experiments.tex` figures and tables. Depends on `peptide_experiment/`,
never the reverse. See its README.
