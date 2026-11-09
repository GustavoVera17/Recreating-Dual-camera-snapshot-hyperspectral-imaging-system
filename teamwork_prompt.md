The user requests a team of auditing, programming, and advisory agents to audit the code and paper, ensuring the code achieves maximum fidelity to what is described in the paper through an iterative feedback loop, reaching 100 points in an automated audit before any future training. For this task, construct the technical plan and the complete codebase in 'C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2', and do not start training yet.

Working directory: C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2
Integrity mode: development

## Reference Material
- Paper: "C:\Users\gustavo.veracoechea\Zotero\storage\P87XBBDG\Xie et al. - 2022 - Dual camera snapshot hyperspectral imaging system via physics-informed learning.pdf" (Xie et al., 2022)
- Previous audit repo: "C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep" (contains analysis of prior defects: Stage 1 Conv2d 1x1 error instead of 1xλ filters, OOM reduction to W8, dataloader frozen seed RandomState(idx), lack of spectral bounding)
- Dataset CAVE: "C:\CODE2026\CASSIproy2026\CASSIproy2026\CAVE_Raw" (or splits in "C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\dataset": 20 train, 10 test)
- Practice and audit reference repo: "C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep_2" (structure for periodic PNG dashboards, CSV with ETA formatted as DD:HH:MM:SS, PyTorch AMP, and automated 100-point audit rubric)
- Active Python environment: `C:\CODE2026\.venv\Scripts\python.exe` (PyTorch with CUDA support)

## Requirements

### R1. Reverse Engineering Plan and Paper Alignment
Develop a technical reverse-engineering specification and architecture plan in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2` that aligns the implementation with Xie et al. (2022) and fixes all identified deficiencies from `Xie_Rep` (Stage 1 filters 1xλ instead of Conv2d 1x1, full-resolution support without OOM via PyTorch AMP/gradient checkpointing, unfrozen dataloader avoiding RandomState(idx), and spectral bounding).

### R2. Codebase Implementation
Build all source code in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2`, including model architecture, forward physics model, CAVE data loading pipeline, loss functions, and training scripts following `Yamawaki_Rep_2` conventions (periodic PNG dashboards, CSV logging with DD:HH:MM:SS ETA, PyTorch AMP).

### R3. Strict 100-Point Audit Rubric
Implement `audit_rubric.py` in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2` providing automated, programmatic scoring totaling 100 points across paper fidelity, architectural correctness, data pipeline integrity, and training harness readiness. Execute an iterative audit-correction loop until `audit_rubric.py` returns an objective score of 100/100 points.

### R4. No Training Execution Constraint
Do not initiate model training during this phase; verify readiness via the audit rubric and dry-run execution checks only.

## Acceptance Criteria

### Technical Specification and Codebase
- [ ] Technical reverse engineering plan is written to `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2`.
- [ ] Complete codebase (model architecture with 1xλ Stage 1 filters, forward model, CAVE dataset loader without frozen random seeds, loss functions, and training script with AMP, CSV ETA in DD:HH:MM:SS format, and PNG dashboard generation) is implemented in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2`.

### Audit Rubric and Verification
- [ ] `audit_rubric.py` is implemented in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2` with an objective 100-point rubric checking paper fidelity, architecture, data loading, and training harness components.
- [ ] Programmatic execution of `python audit_rubric.py` in `C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep_2` outputs an audit score of exactly 100/100 points.
- [ ] No model training run or background training process has been initiated.
