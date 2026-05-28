# Data Risks

## Current Status

- Real LIBERO demonstration inspected: `True`
- Dataset root: `/home/zhan_shaoji/data/libero/datasets`
- Demonstration path: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5`
- Inspection report: `results/inspections/20260527_093521_libero_data_inspection_real.json`
- Previous G1.5 state before real-data inspection: `not observed` / `blocked by G1.5`.

## Risk Register

| Risk | Status | Notes |
| --- | --- | --- |
| Real demonstration file path unknown | resolved | Must be recorded before dataset implementation. |
| Real action dimension unknown | resolved | Do not assume action dimension before one real HDF5 demonstration is inspected. |
| Camera keys unknown | resolved | Must inspect real image/camera keys before choosing inputs. |
| State/proprio keys unknown | resolved | Must identify real state/proprio fields before deciding whether state is input or audit-only. |
| Language keys unknown | resolved | Instruction observed at `attrs/data/problem_info/language_instruction`. |
| Split format unknown | policy defined | `docs/SPLIT_POLICY.md` defines Phase-1 trajectory-level split policy; real loader still pending. |
| Action alignment unknown | resolved with risks | `docs/LIBERO_ACTION_SEMANTICS.md` documents `action_to_current_obs`; fresh replay validation is still future work. |
| Future leakage through inputs | resolved for v1 | Synthetic trajectory-window tests cover implemented image/action/state/future-frame fields. |
| Split leakage | policy defined | `docs/NORMALIZATION_POLICY.md` and tests enforce train-only normalization helpers; full real loader still pending. |

## Do Not Proceed

Do not proceed to WAM-style future-latent claims until `target_future_latents` is implemented and tested. Action-only training implementation may begin only if it follows `docs/LIBERO_ACTION_SEMANTICS.md`, `docs/SPLIT_POLICY.md`, and `docs/NORMALIZATION_POLICY.md`.
