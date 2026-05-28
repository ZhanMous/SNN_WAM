# Data Risks

## Current Status

- Real LIBERO demonstration inspected: `True`
- Dataset root: `/home/zhan_shaoji/data/libero/datasets`
- Demonstration path: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`
- Inspection report: `results/inspections/20260528_033539_libero_data_inspection_real.json`
- Previous G1.5 state before real-data inspection: `not observed` / `blocked by G1.5`.

## Risk Register

| Risk | Status | Notes |
| --- | --- | --- |
| Real demonstration file path unknown | resolved | Must be recorded before dataset implementation. |
| Real action dimension unknown | resolved | Do not assume action dimension before one real HDF5 demonstration is inspected. |
| Camera keys unknown | resolved | Must inspect real image/camera keys before choosing inputs. |
| State/proprio keys unknown | resolved | Must identify real state/proprio fields before deciding whether state is input or audit-only. |
| Language keys unknown | resolved | Must identify real instruction field source. |
| Split format unknown | unresolved | Must inspect official file layout and split metadata before defining train/val/test policy. |
| Action alignment unknown | resolved | Use `docs/LIBERO_ACTION_SEMANTICS.md`; fresh replay validation remains future work. |
| Future leakage through inputs | partially resolved | Synthetic trajectory-window tests cover action, image, and dry-run future-latent alignment; real frozen latent extraction remains future work. |
| Split leakage | unresolved | Requires train/val/test split policy and no normalization on val/test. |

## Do Not Proceed

Do not proceed to real-data WAM-style future-latent claims until frozen visual latents are precomputed or adapter-produced with recorded metadata. Offline dry-run WAM training may run only as a smoke test and must follow `docs/LIBERO_ACTION_SEMANTICS.md`, `docs/SPLIT_POLICY.md`, and `docs/NORMALIZATION_POLICY.md`.
