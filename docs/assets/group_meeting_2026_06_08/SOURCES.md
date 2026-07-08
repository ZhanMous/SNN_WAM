# Group Meeting 2026-06-08 Slide Sources

This directory contains visual assets used by:

- `docs/GROUP_MEETING_2026-06-08_THREE_WEEK_PROGRESS_SLIDES.md`
- `docs/GROUP_MEETING_2026-06-08_THREE_WEEK_PROGRESS.pdf`

## Official Literature Figures

| Local file | Source | Use in slides | Notes |
|---|---|---|---|
| `dino_wm_intro.png` | `https://dino-wm.github.io/mfiles/figs/intro.png` | Slide 4, DINO-WM overview | Official DINO-WM project figure. Literature explanation only. |
| `dino_wm_model_arch.png` | `https://dino-wm.github.io/mfiles/figs/model_arch.png` | Slide 5, DINO-WM architecture / planning loop | Official DINO-WM project figure. The deck does not claim local DINO-WM reproduction. |
| `eggroll_diagram.png` | `https://eshyperscale.github.io/imgs/diagram.png` | Slide 6, EGGROLL low-rank perturbation | Official HyperscaleES / EGGROLL project figure. |
| `eggroll_header.png` | `https://eshyperscale.github.io/blog/header.png` | Downloaded reference asset; not used in the current 16-slide PDF | Official HyperscaleES / EGGROLL project figure. It is not a SNN-WAM result. |

## Official Text / Citation Sources

| Source | Use |
|---|---|
| `https://dino-wm.github.io/` | DINO-WM literature framing: DINOv2 spatial patch features, future patch prediction, visual goal planning. |
| `https://github.com/gaoyuezhou/dino_wm` | Official DINO-WM code/repo source. |
| `https://arxiv.org/abs/2411.04983` | DINO-WM paper metadata and citation. |
| `https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/` | DINOv2 background: self-supervised features and pretrained visual backbone. |
| `https://github.com/facebookresearch/dinov2` | DINOv2 backbone names and usage source. |
| `https://eshyperscale.github.io/` | HyperscaleES / EGGROLL project explanation and official figures. |
| `https://arxiv.org/abs/2511.16652` | Evolution Strategies at the Hyperscale paper metadata and citation. |
| `https://github.com/ESHyperscale/HyperscaleES` | Official HyperscaleES code source. |
| `https://github.com/ESHyperscale/nano-egg` | nano-egg official code source. |

## Self-Drawn Figures

The following slide visuals are self-drawn with HTML/CSS in the Marp source:

- Slide 2: old action-BC route vs current latent world-model route.
- Slide 3: DINOv2 frame-to-patch-latent schematic.
- Slide 7: nano-egg role schematic.
- Slide 8: DINO-WM / EGGROLL / SNN-WAM research-connection schematic.
- Slide 9: three-workstream status matrix.
- Slide 10: closed-loop diagnostic bar chart.
- Slide 11: new SNN-WAM route pipeline.
- Slide 12: DWM gate status matrix.
- Slide 13: real DWM-G3 persistence comparison bar chart.
- Slide 14: HyperscaleES reproduction config and completed-run matrix.
- Slide 15: HyperscaleES generation-quality degeneration timeline.
- Slide 16: risks and stop rules.

These self-drawn figures summarize the project route and local evidence; they are not copied from papers.

## Local Evidence Sources

| Slide | Evidence source | Evidence status |
|---|---|---|
| 3 | `docs/CLAIMS_LEDGER.md`, claim `C-DWM-G1-001`; `docs/RESULT_ARTIFACTS.md`, artifact `R-DWM-G1-001` | Gate validation, not performance evidence. |
| 8 | `docs/CLAIMS_LEDGER.md`, claims `C-DWM-ROUTE-001` and `C-DWM-ES-ROLE-001` | Hypothesis / route-positioning only. |
| 9 | `docs/GROUP_MEETING_2026-06-08_THREE_WEEK_PROGRESS.md`; user-provided HyperscaleES reproduction notes, 2026-06-08 | Progress summary; ES evidence is external engineering / diagnostic. |
| 10 | `docs/CLAIMS_LEDGER.md`, claims `C-G5-EVALUATOR-VALIDITY-001` and `C-G5-WAM-GRU-FAILURE-001`; artifact `R-G5-DIAGNOSTIC-EVAL-001` | Supported diagnostic evidence. Expert replay 27/30; WAM-GRU future/no-future 0/30. |
| 12 | `docs/CLAIMS_LEDGER.md`, claims `C-DWM-G1-001`, `C-DWM-G2-001`, `C-DWM-G3-001`, `C-DWM-G3-REAL-DIAG-001` | G1/G2 supported gate validation; G3 synthetic observation; real G3 preliminary diagnostic. |
| 13 | `docs/RESULT_ARTIFACTS.md`, artifact `R-DWM-G3-DINOWM-BASELINE-REAL-001`; `results/runs/dinowm_transformer_baseline_real/eval_multihorizon/eval_metrics.csv`; `results/runs/dinowm_transformer_baseline_real/baselines/persistence_metrics.json` | Preliminary / diagnostic only. H1 model 0.02808 vs persistence 0.02719; H4 model 0.19858 vs persistence 0.04768. |
| 14 | User-provided HyperscaleES reproduction notes, 2026-06-08 | External engineering diagnostic notes only; not registered SNN-WAM scientific evidence. The slide must not be read as successful paper reproduction. |
| 15 | User-provided HyperscaleES reproduction notes, 2026-06-08 | External generation-quality diagnostic. Output snippets are shortened for presentation. |
| 16 | `docs/CLAIMS_LEDGER.md` forbidden claims and gate notes; user-provided HyperscaleES reproduction notes, 2026-06-08 | Claim-boundary / risk slide. ES stop rule is based on observed generation collapse. |

## User-Provided Diagnostic Notes

The RWKV-7 1.5B HyperscaleES / EGGROLL reproduction details on slides 9, 14, 15, and 16 come from user-provided meeting notes dated 2026-06-08. They describe external engineering diagnostics, not SNN-WAM result artifacts. Allowed wording:

- "Multiple RWKV-7 1.5B GSM8K/Countdown EGGROLL runs were attempted."
- "Generation quality degraded during training."
- "These runs do not support a useful-optimization claim."

Forbidden wording: avoid any phrasing that upgrades these diagnostics into a successful paper reproduction, useful RWKV-7 math fine-tuning result, or evidence that ES trains the SNN world model.

## Font

| Local file | Source | Use |
|---|---|---|
| `NotoSansCJKsc-Regular.otf` | `https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf` | Chinese PDF rendering. |
| `NotoSansCJKsc-Bold.otf` | `https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf` | Chinese PDF rendering. |
