# TFHE Boolean parameter sets

JSON files consumed by `bench_tfhe --params=<file>` (see `btc::load_boolean_parameters`
in `include/btc/tfhe_params_io.hpp`). Each file specifies every field of the C API's
`BooleanParameters` struct explicitly — no preset is implied, so results stay
reproducible and comparable across schemes.

## Fields

| Field                     | Meaning                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `lwe_dimension`            | LWE secret key size. Bigger = more secure, slower bootstrap.            |
| `glwe_dimension`           | GLWE dimension (used in the bootstrapping key).                         |
| `polynomial_size`          | Ring polynomial degree for GLWE. Bigger = more secure, slower PBS.      |
| `lwe_noise_distribution`   | Noise added at LWE encryption. `{ "gaussian_std_dev": <double> }` or `{ "t_uniform_bound_log2": <int> }`. |
| `glwe_noise_distribution`  | Noise added at GLWE (bootstrapping key) level. Same shape as above.     |
| `pbs_base_log` / `pbs_level` | Decomposition params for programmable bootstrapping (the expensive gate op). More levels = lower noise growth, more compute per gate. |
| `ks_base_log` / `ks_level`   | Decomposition params for key switching. Same base/level trade-off.    |
| `encryption_key_choice`    | `"big"` or `"small"` — whether bootstrap happens before or after key switch, shifting where the cost sits. |

## Files

- `default_parameters.json` — replica of TFHE-rs' `BOOLEAN_PARAMETERS_SET_DEFAULT_PARAMETERS`
  (132-bit security, p-fail ~2^-64.34).
- `default_parameters_ks_pbs.json` — replica of `BOOLEAN_PARAMETERS_SET_DEFAULT_PARAMETERS_KS_PBS`
  (same security target, KS-then-PBS ordering).
- `template.json` — copy of `default_parameters.json` to fork from when sweeping a
  single field (e.g. only vary `pbs_level` while holding the rest fixed).

## Usage

```bash
./build/benchmarks/bench_tfhe graph_N8 --params=data/tfhe_params/default_parameters.json
```

Omit `--params` to fall back to `boolean_gen_keys_with_default_parameters` (unchanged
prior behavior). The graph name and `--params=` flag can be given in either order.
