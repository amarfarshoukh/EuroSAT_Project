# AGENTS.md — EuroSAT Deep Learning Project

## Project purpose
Build a complete, reproducible Jupyter-notebook implementation for the Master 2 Programming for AI project:
**Satellite Image Land-Use Classification on the EuroSAT Dataset**.

The implementation must follow:
1. `docs/M2_AI_Project_Assignment2026-2027.pdf`
2. `docs/Our plan.pdf`
3. The instructor's course/lab notebooks in `docs/course_labs/`
4. The existing student notebook `notebooks/EuroSAT_Project.ipynb` as a reference only.

## Critical file rule
- NEVER overwrite, rename, or edit `notebooks/EuroSAT_Project.ipynb`.
- Create the new implementation as:
  `notebooks/EuroSAT_Codex_Implementation.ipynb`
- Keep the original notebook available for comparison.

## Environment constraints
- Target environment: Windows laptop, CPU only.
- Do NOT assume CUDA/GPU availability.
- Use PyTorch / torchvision.
- Prefer CPU-practical models and experiment settings.
- Use `num_workers=0` for DataLoaders unless there is a demonstrated reason to change it.
- Avoid unnecessary 224x224 scratch-CNN training.
- Transfer learning must remain CPU-conscious.

## Notebook-first requirement
The notebook is the primary artifact. Do not hide the core project logic in many external `.py` files.
Reusable helper functions are encouraged, but students must be able to see, run, explain, and modify the important code directly in the notebook.

The notebook must run top-to-bottom without undefined variables or hidden execution-order dependencies.

## Instructor-style coding
Follow the style used in the instructor's PyTorch labs:
- clear sequential Jupyter cells;
- short Markdown explanation before every code cell;
- short Markdown explanation after every important code cell;
- `torchvision.transforms.Compose(...)` for preprocessing/augmentation;
- `Dataset` / `DataLoader`;
- explicit tensor shapes and device handling;
- models subclass `nn.Module`;
- architecture defined in `__init__`, computation in `forward`;
- explicit loss and optimizer definitions;
- standard training order:
  `optimizer.zero_grad()` -> forward -> loss -> `loss.backward()` -> `optimizer.step()`;
- use `model.train()` for training;
- use `model.eval()` and `torch.no_grad()` for validation/test;
- use Matplotlib for figures;
- save/load checkpoints with `state_dict`;
- keep code readable enough for a live oral defense.

Do not over-engineer with complex classes, frameworks, decorators, callbacks, or abstractions that are not needed for the assignment.

## Explanation rule
Before EACH code cell, insert a brief Markdown cell:
- heading;
- 1–3 sentences saying what the code does and why.

After EACH important code cell, insert a brief Markdown cell:
- `**What to understand:**`
- 1–3 bullets explaining the key concept(s);
- if the cell produces results, explain what should be inspected, but NEVER invent a result that has not been run.

Keep explanations concise and student-friendly.

## Reproducibility
Use a fixed random seed consistently (default `SEED = 42`):
- Python `random`
- NumPy
- PyTorch

Use one fixed train/validation/test split for all controlled scratch-CNN experiments.
Default split from the team plan:
- 70% train
- 15% validation
- 15% test

All architecture/hyperparameter/augmentation comparisons must use the SAME split.

## Data rules
Dataset: `torchvision.datasets.EuroSAT`.

10 classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Keep dataset files under `data/` and out of Git.

For the scratch-CNN pipeline:
- keep EuroSAT at 64x64 unless there is an explicit documented experiment;
- use a documented normalization strategy consistently;
- if using `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`, correctly unnormalize images before visualization;
- validation and test transforms must not contain random augmentation.

For pretrained ResNet-18:
- use ImageNet-compatible normalization;
- choose an image size that is technically valid and CPU-practical;
- document the choice clearly.

## Test-set discipline
- Use validation data for selecting architectures, optimizer/LR, scheduler, and augmentation.
- Never select a configuration based on test accuracy.
- The assignment requires baseline diagnostics on unseen test data, so baseline test evaluation may be reported, but it must NOT guide later choices.
- Final scratch and transfer-learning models must be evaluated on the test set.

## Stage structure

### Stage 1 — Common dataset pipeline and baseline CNN
Must contain:
- dataset download/load;
- class list;
- fixed train/val/test split;
- train/val/test DataLoaders;
- sample visualization;
- baseline CNN from scratch;
- parameter count;
- loss + optimizer;
- epoch-by-epoch training loss;
- epoch-by-epoch validation loss;
- epoch-by-epoch validation accuracy;
- per-class accuracy;
- learning curves;
- normalized confusion matrix;
- checkpoint save/load;
- inference on unseen mini-batch;
- baseline result summary.

Required key checkpoint:
`checkpoints/baseline.pth`

### Stage 2 — Architecture experiments
Controlled experiments:
- depth: 2 vs 3 vs 4 convolutional layers;
- kernel size: 3x3 vs 5x5 vs 7x7;
- BatchNorm comparison;
- controlled stride/padding variation.

Change ONE intended factor at a time.
Keep the remaining settings fixed.

For every run record:
- experiment ID;
- architecture/change;
- total parameters;
- trainable parameters;
- epochs;
- batch size;
- peak validation accuracy;
- final training accuracy;
- final validation accuracy;
- validation loss;
- runtime;
- inference latency;
- generalization gap;
- checkpoint filename;
- notes.

Do not use test accuracy to choose the Stage 2 winner.

### Stage 3 — Optimizer and learning-rate experiments
Use the selected architecture from Stage 2.

Required controlled runs:
- SGD + Momentum: 1e-2, 1e-3, 1e-4
- Adam: 1e-2, 1e-3, 1e-4
- AdamW: 1e-2, 1e-3, 1e-4

Then compare the selected optimizer/LR:
- without scheduler;
- with one sensible scheduler.

Do not fabricate the selected configuration before experiments are run.
Use a clearly marked configuration cell that the student updates after reviewing actual results.

### Stage 4 — Data augmentation experiments
Keep selected architecture + optimizer + LR fixed.

Control:
- no augmentation.

Required comparisons:
- random flips + random crop;
- affine transformations + ColorJitter;
- MixUp OR CutMix.

Use instructor-style `torchvision.transforms` where applicable.
Clearly separate training transforms from validation/test transforms.

### Stage 5 — Final optimized scratch CNN
Combine only choices selected from actual Stage 2–4 validation results.

Required checkpoint:
`checkpoints/best_scratch_cnn.pth`

Generate:
- training/validation loss curves;
- training/validation accuracy curves;
- per-class accuracy;
- normalized confusion matrix;
- final test accuracy;
- parameter count;
- runtime / latency;
- generalization gap.

### Stage 6 — Transfer learning
Backbone: ResNet-18 pretrained on ImageNet unless the source documents require a different choice.

Experiments:
1. Frozen feature extractor — new classifier head trainable only.
2. Partial fine-tuning — classifier + selected upper block(s).
3. Full fine-tuning — entire network trainable with reduced LR.

Required checkpoints:
- `checkpoints/resnet_frozen.pth`
- `checkpoints/resnet_partial_finetuned.pth`
- `checkpoints/resnet_full_finetuned.pth`

Show `requires_grad` logic clearly and count trainable parameters.

### Stage 7 — Final evaluation and comparison
Compare:
- baseline scratch CNN;
- best optimized scratch CNN;
- frozen ResNet-18;
- partially fine-tuned ResNet-18;
- fully fine-tuned ResNet-18.

Include:
- final test metrics;
- convergence behavior;
- data efficiency;
- parameter counts;
- trainable parameters;
- runtime;
- inference latency;
- generalization gap;
- per-class accuracy;
- normalized confusion matrices.

### Stage 8 — Results export / report support
Create clean tables/CSV outputs in `results/` that can be reused in the report and presentation.
Do not fabricate values.

### Stage 9 — Checkpoint verification
Provide cells that:
- create fresh model instance;
- load saved `state_dict`;
- move to CPU device;
- switch to eval mode;
- run inference on unseen test mini-batch;
- display predicted vs true labels.

This must be simple enough to demonstrate live.

### Stage 10 — Defense preparation cells
Include a compact section demonstrating easy modifications:
- kernel size;
- activation;
- optimizer / LR;
- augmentation;
- freeze/unfreeze a ResNet block;
- load checkpoint and infer.

Do not retrain the whole project in this section.

## Experiment logging
Maintain a single results table and save it as:
`results/experiment_results.csv`

At minimum include columns:
`experiment_id, stage, model_name, architecture_change, kernel_size, stride, padding, batchnorm, optimizer, learning_rate, scheduler, augmentation, epochs, batch_size, total_params, trainable_params, peak_val_accuracy, final_train_accuracy, final_val_accuracy, final_val_loss, test_accuracy, runtime_seconds, inference_latency_ms, generalization_gap, checkpoint, notes`

Use `NaN`/blank where a field is not appropriate. Never invent values.

## Runtime / CPU rules
This project is developed on CPU.
- Provide a `FAST_MODE` development option for smoke tests.
- FAST_MODE must never silently replace final experiment settings.
- Long experiment cells must be clearly labeled.
- Do not launch all full experiments automatically while generating the notebook.
- During implementation validation, run only lightweight smoke tests (for example one mini-batch or at most 1 short epoch) unless the user explicitly requests long training.
- Full runs will be launched manually by the student.

## Fair comparison rules
When testing one factor, keep important non-tested settings constant.
Reset/reinitialize model weights for each independent run.
Do not reuse trained weights across scratch architecture/hyperparameter comparisons unless the experiment explicitly requires it.

## Checkpoint rules
Use clear names. No names like `final_final2.pth`.
Save state dictionaries, not pickled whole model objects.
Checkpoint directory: `checkpoints/`.

## Code quality rules
- readable variable names;
- concise inline comments;
- no duplicated huge training loops when a clear reusable function can avoid duplication;
- no hidden global state when a function parameter is clearer;
- no undefined cells;
- no fake metrics;
- no hard-coded GPU calls;
- no absolute machine-specific paths;
- use project-relative paths;
- Windows-safe code.

## Git rules
Never add:
- `data/`
- `checkpoints/`
- `.ipynb_checkpoints/`
- virtual environments
to Git.

Notebook, source, results CSV/tables, documentation, and `.gitignore` should remain version-controlled.

## Validation before completion
Before claiming the notebook is ready:
1. Verify notebook JSON is valid.
2. Verify all imports.
3. Run lightweight smoke tests on CPU.
4. Verify one forward pass produces shape `[batch_size, 10]` for scratch model.
5. Verify checkpoint save/load round-trip.
6. Verify results directory creation.
7. Verify no test data is used for model selection.
8. Verify the original `EuroSAT_Project.ipynb` was not modified.
9. List anything that still requires a real full training run.

## Important academic rule
This notebook is for a live oral defense. Favor code that the students can understand and explain.
If a clever abstraction makes the code harder to defend, prefer the simpler explicit implementation.
