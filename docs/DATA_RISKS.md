# Data Risks

## Current Status

- Real LIBERO demonstration inspected: `True`
- Dataset root: `/home/zhan_shaoji/data/libero/datasets`
- Demonstration path: `/home/zhan_shaoji/data/libero/datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`
- Inspection report: `results/inspections/20260528_020959_libero_data_inspection_real.json`
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
| Future leakage through inputs | unresolved | Requires future synthetic trajectory-window tests before G2 passes. |
| Split leakage | unresolved | Requires train/val/test split policy and no normalization on val/test. |

## Do Not Proceed

Do not proceed to WAM-style future-latent claims until `target_future_latents` is implemented and tested. Action-only training implementation may begin only if it follows `docs/LIBERO_ACTION_SEMANTICS.md`, `docs/SPLIT_POLICY.md`, and `docs/NORMALIZATION_POLICY.md`.
