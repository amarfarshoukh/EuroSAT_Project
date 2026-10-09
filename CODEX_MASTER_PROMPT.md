# Master Prompt for Codex — EuroSAT Project

You are working inside my local repository for a Master 2 Programming for AI project.

Before editing anything:
1. Read the repository-root `AGENTS.md` completely and follow it.
2. Read:
   - `docs/M2_AI_Project_Assignment2026-2027.pdf`
   - `docs/Our plan.pdf`
3. Inspect the instructor course materials in:
   - `docs/course_labs/`
   especially the PyTorch tensors/autograd/neural-network labs, the Transforms & DataLoaders lab, the Train-a-Classifier lab, and the Data Augmentation lab.
4. Inspect `notebooks/EuroSAT_Project.ipynb` to understand what I already implemented.
5. DO NOT modify or overwrite `notebooks/EuroSAT_Project.ipynb`.

If any PDF cannot be inspected reliably, do not guess what it contains. Say which file/page could not be read and rely on the explicit rules in `AGENTS.md`.

## Main task
Create a second, independent notebook:

`notebooks/EuroSAT_Codex_Implementation.ipynb`

The purpose of this notebook is:
- to implement the FULL assignment structure from Stage 1 through Stage 10;
- to be properly coded and reproducible;
- to remain CPU-only and practical on my laptop;
- to follow the instructor's coding style as closely as possible;
- to let me compare it with my existing notebook;
- to help me study every part before the oral defense.

Do not delete or change my existing notebook.

## Very important notebook-writing style
For EVERY code cell:
1. Put a short Markdown cell immediately BEFORE it explaining:
   - what this code is about to do;
   - why it is needed.
2. For every important code cell, put a short Markdown cell immediately AFTER it titled:
   `**What to understand:**`
   with 1–3 brief bullets explaining the important concept or what output to inspect.

Keep explanations brief. I do not want long textbook paragraphs.

Examples of the tone:
- "This cell creates the fixed train/validation/test split used by every controlled experiment."
- "What to understand: `shuffle=True` is used only for training; validation/test order does not need shuffling."
- "What to understand: `loss.backward()` computes gradients; `optimizer.step()` updates the parameters."

Do NOT write fake interpretations such as "accuracy improved to 90%" unless the notebook has actually been executed and produced that result.

## Required implementation strategy
Build the notebook in clear numbered sections:

### 0. Project Setup
- imports;
- versions;
- seed;
- CPU device;
- project paths;
- reproducibility.

### 1. Stage 1 — Dataset and Baseline
Implement the complete common EuroSAT pipeline and baseline from scratch.

### 2. Stage 2 — Architecture Experiments
Prepare controlled experiments for:
- depth 2/3/4 conv layers;
- kernel 3/5/7;
- BatchNorm;
- stride/padding.

Create reusable experiment functions so we do not copy the same large training loop repeatedly.

### 3. Stage 3 — Optimizer and Learning Rate
Prepare all 9 required optimizer/LR combinations:
- SGD+Momentum x 1e-2, 1e-3, 1e-4;
- Adam x 1e-2, 1e-3, 1e-4;
- AdamW x 1e-2, 1e-3, 1e-4.

Then prepare a scheduler comparison.

IMPORTANT:
Do not invent which architecture or optimizer is best.
Create clearly marked selection/config cells that I update AFTER I run experiments and review the validation results.

### 4. Stage 4 — Augmentation
Prepare:
- no augmentation control;
- flip + random crop;
- affine + ColorJitter;
- MixUp or CutMix.

Keep validation/test transforms deterministic.

### 5. Stage 5 — Final Scratch CNN
Make this depend on the configuration selected from actual Stage 2–4 results.
Do not hard-code a fake winner.

### 6. Stage 6 — Transfer Learning
Use pretrained ResNet-18 and implement:
- frozen feature extractor;
- partial fine-tuning;
- full fine-tuning with reduced LR.

Use ImageNet normalization.
Keep the input-size choice CPU-conscious and explain it.

### 7. Stage 7 — Final Comparison
Create all required figures and comparison tables.

### 8. Stage 8 — Results Export
Save structured results to:
`results/experiment_results.csv`

### 9. Stage 9 — Checkpoint Loading / Live Inference
Make checkpoint loading extremely clear and easy to demonstrate live.

### 10. Stage 10 — Defense Modification Examples
Add a compact final section that shows where/how to:
- change a kernel;
- change activation;
- change optimizer/LR;
- change augmentation;
- freeze/unfreeze a ResNet block;
- reload a checkpoint and infer.

## Training execution
DO NOT start all long training jobs while creating the notebook.

Create:
`FAST_MODE = True`

When `FAST_MODE=True`, use only lightweight smoke-test settings so we can validate the pipeline.
When `FAST_MODE=False`, use the documented real experiment settings.

Clearly show the real settings. Never let FAST_MODE silently change what gets reported as final experimental results.

Run enough smoke tests to prove that:
- dataset loads;
- loaders work;
- scratch model forward pass works;
- training function can complete a tiny run;
- evaluation function works;
- checkpoint save/load works.

Stop before expensive full experiment sweeps.

## Results integrity
Never fabricate metrics.
Never fill result tables with invented numbers.
Use validation data to select experiment winners.
Do not select models based on test accuracy.

Where a later stage needs a winner from an earlier stage, use a clearly visible placeholder/configuration cell and explain exactly what I must fill after running experiments.

## Baseline / test-set nuance
The assignment requires baseline diagnostics including a normalized confusion matrix on unseen test data, so baseline test reporting is allowed.
However, baseline test results must NOT be used to choose later architectures/hyperparameters.

## Code requirements
Use simple PyTorch code that resembles the instructor labs:
- `nn.Module`;
- explicit `forward`;
- `CrossEntropyLoss`;
- `torch.optim`;
- standard training loop;
- `model.train()`;
- `model.eval()`;
- `torch.no_grad()`;
- `state_dict`;
- Matplotlib;
- torchvision transforms and DataLoaders.

Do not introduce PyTorch Lightning, Hugging Face Trainer, complex callback frameworks, or unnecessary third-party experiment trackers.

## Deliverables from you
When finished:
1. Create `notebooks/EuroSAT_Codex_Implementation.ipynb`.
2. Create/update `results/experiment_results.csv` schema if needed, but do not invent rows with fake metrics.
3. Do not touch the original notebook.
4. Run lightweight CPU smoke tests only.
5. Give me a concise completion report containing:
   - files created/changed;
   - smoke tests run;
   - what passed;
   - what full training still needs to be run manually;
   - any assumptions you made;
   - any places where the instructor PDF could not be reliably read.

Before making edits, first summarize your planned notebook structure in 10–15 lines. Then proceed with implementation.
