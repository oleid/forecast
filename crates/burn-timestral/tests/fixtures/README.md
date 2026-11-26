# Timestral parity fixtures

5M, 28M, and 123M use the same three reference scenarios: left padding for a context length that is not a multiple of the patch size, a three-block autoregressive rollout with a known covariate, and past and known numeric covariates with NaN gaps. The 209M variant retains all 17 golden scenarios. Each parity test loads its checkpoint once and compares all forecast quantiles with the Python reference (`atol = rtol = 1e-4`).

### Rust tests

In CI, each model job generates fresh reference quantiles with CPU PyTorch and
compares them with Rust/Flex on the same runner using the same pinned checkpoint
files. The generated JSON is passed via `TIMESTRAL_PARITY_FIXTURE`; set it only
when running a single variant. Without that override the model tests use the
checked-in fixtures. The offline integrity test always checks the checked-in files.

Without a checkpoint (only fixture metadata and integrity checks; the four model tests are ignored by default):

```sh
cargo test --package burn-timestral --no-default-features --test test_inference_parity
```

Run all four through Hub support, enabled by default. This uses the pinned checkpoints from the cache or downloads them:

```sh
unset TIMESTRAL_MODEL_DIR
cargo test --package burn-timestral --release --test test_inference_parity -- --ignored --test-threads=1 --nocapture
```

The parity tests use Burn's default device (Flex in a standard build). To run the same comparison on WGPU, enable `wgpu` and use Burn's `BURN_DEVICE` override to select it explicitly:

```sh
BURN_DEVICE=wgpu cargo test --locked --package burn-timestral --release --features wgpu --test test_inference_parity test_parity_5m -- --ignored --exact --nocapture
```

WGPU requires a compatible adapter. To force Flex even with WGPU enabled, set `BURN_DEVICE=flex`.

With `hub` enabled, the test uses the pinned checkpoint from the Hugging Face cache or downloads it if needed. Set `TIMESTRAL_MODEL_DIR` only to use a checkpoint from a different local directory containing the original `config.json` and `model.safetensors` files. This override applies to the whole test process, so run exactly one test when using it:

```sh
TIMESTRAL_MODEL_DIR=target/parity-models/5M cargo test --package burn-timestral --release --test test_inference_parity test_parity_5m -- --ignored --exact --nocapture
TIMESTRAL_MODEL_DIR=target/parity-models/28M cargo test --package burn-timestral --release --test test_inference_parity test_parity_28m -- --ignored --exact --nocapture
TIMESTRAL_MODEL_DIR=target/parity-models/123M cargo test --package burn-timestral --release --test test_inference_parity test_parity_123m -- --ignored --exact --nocapture
TIMESTRAL_MODEL_DIR=~/models/Timestral-209M cargo test --package burn-timestral --release --test test_inference_parity test_parity_209m -- --ignored --exact --nocapture
```

### Regenerate the Python reference fixtures offline

Use a clean `mistralai/timestral` checkout at `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`, Python 3.12, PyTorch 2.15.0 (CPU), and NumPy 2.5.5. Adjust the paths as needed. The generator retains `parity_scenarios.json` for 209M and uses `parity_scenarios_<variant>.json` for the smaller variants.

```sh
(
    checkout=~/.cache/uv/git-v2/checkouts/d3d88cfebe3e1c5e/a534bdb
    for variant in 5M 28M 123M 209M; do
        case "$variant" in
            209M) model=~/models/Timestral-209M ;;
            *) model=target/parity-models/$variant ;;
        esac
        UV_OFFLINE=1 uv run --offline --no-project --python 3.12 \
            --with 'torch==2.15.0' --with 'numpy==2.5.5' --with "$checkout" \
            python crates/burn-timestral/tools/generate_parity_fixtures.py --variant "$variant" \
            --checkpoint "$model" --timestral-checkout "$checkout" || exit
    done
)
```

Official checkpoint revisions (Hub repositories `mistralai/Timestral-<variant>`):

| Variant | Revision |
| --- | --- |
| 5M | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| 28M | `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| 123M | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| 209M | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

Each fixture records the SHA-256 hashes of the local weights and configuration and the measured Git revision of the Python reference. `--checkpoint` must point to the original files from the listed Hub revision. The generator records the local hashes but does not verify them against the Hub. If provided, `--timestral-checkout` must identify the checkout used to install the Python reference.