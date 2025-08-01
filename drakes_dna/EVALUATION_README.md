# DRAKES DNA Model Evaluation System

This evaluation system converts the functionality from `eval.ipynb` into a comprehensive Python script with proper logging, statistics saving, and YAML configuration support.

## Features

✅ **Comprehensive Evaluation Pipeline**:
- Log-likelihood calculation
- Oracle predictions (HepG2, K562, SKNSH)
- ATAC-seq accessibility predictions
- 3-mer Pearson correlation analysis
- JASPAR motif analysis

✅ **Proper Logging**: All operations logged with timestamps
✅ **Structured Output**: Results saved in multiple formats (JSON, CSV, pickle)
✅ **YAML Configuration**: Easy configuration management
✅ **Multiple Model Support**: Evaluate various model checkpoints
✅ **Flexible Sampling**: Different sampling methods (TDS, CG, SMC, CFG)

## Files

- **`eval_models.py`**: Main evaluation script
- **`eval_config.yaml`**: Configuration file template
- **`run_evaluation.py`**: Simple runner script
- **`EVALUATION_README.md`**: This documentation

## Quick Start

### 1. Basic Usage
```bash
# Run with default configuration
python run_evaluation.py

# Or run directly with custom config
python eval_models.py eval_config.yaml --output-dir /path/to/output
```

### 2. Configuration

Edit `eval_config.yaml` to specify your model checkpoints:

```yaml
# Base paths
base_path: "/path/to/your/data_and_model"

# Model checkpoints to evaluate
models:
  finetuned_model: "mdlm/reward_bp_results_final/finetuned.ckpt"
  pretrained_model: "mdlm/outputs_gosai/pretrained.ckpt"
  zero_alpha_model: "mdlm/reward_bp_results_final/zero_alpha.ckpt"
  cfg_model: "mdlm/outputs_gosai/cfg.ckpt"

# Sampling parameters
num_sample_batches: 10
num_samples_per_batch: 64
```

### 3. Output Structure

The evaluation creates a timestamped output directory with:

```
evaluation_results_YYYYMMDD_HHMMSS/
├── evaluation_summary.csv          # Main results table
├── evaluation_results.json         # Complete results (human-readable)
├── evaluation_results.pkl          # Complete results (Python objects)
├── motif_summary.csv               # JASPAR motif counts
├── motif_correlations.csv          # Motif correlation matrix
└── evaluation_YYYYMMDD_HHMMSS.log  # Detailed execution log
```

## Detailed Usage

### Command Line Interface

```bash
# Basic usage
python eval_models.py config.yaml

# With custom output directory
python eval_models.py config.yaml --output-dir /custom/path

# Help
python eval_models.py --help
```

### Configuration Options

#### Model Configuration
```yaml
models:
  finetuned_model: "path/to/finetuned.ckpt"     # Your main model
  pretrained_model: "path/to/pretrained.ckpt"   # Baseline pretrained model
  zero_alpha_model: "path/to/zero_alpha.ckpt"   # Zero alpha variant
  cfg_model: "path/to/cfg.ckpt"                 # Classifier-free guidance model
```

#### Sampling Parameters
```yaml
num_sample_batches: 10        # Number of batches to sample
num_samples_per_batch: 64     # Samples per batch
likelihood_steps: 128         # Steps for likelihood calculation
```

#### Controlled Sampling (for pretrained model)
```yaml
use_regular_sampling: true    # Standard sampling
use_tds_sampling: false       # Twisted Diffusion Sampling
use_cg_sampling: false        # Classifier Guidance
use_smc_sampling: false       # Sequential Monte Carlo

# TDS parameters
tds_alpha: 0.5
tds_guidance_scale: 1000

# CG parameters  
cg_guidance_scale: 300000

# SMC parameters
smc_alpha: 0.5
```

### Results Interpretation

#### Main Results (`evaluation_summary.csv`)
| Column | Description |
|--------|-------------|
| `model` | Model name |
| `median_log_likelihood` | Median log-likelihood (higher = better) |
| `median_hepg2_prediction` | Median HepG2 activity prediction |
| `atac_accessibility_rate` | Fraction of sequences with ATAC accessibility > 0.5 |
| `kmer_pearson_correlation` | 3-mer correlation with high-expression sequences |

#### Detailed Results (`evaluation_results.json`)
- Complete statistical distributions
- All predictions and likelihoods
- Motif analysis results
- Metadata and configuration used

## Integration Examples

### Python Integration
```python
from eval_models import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator("my_config.yaml", "output_dir")

# Run full evaluation
evaluator.run_evaluation()

# Access results
results = evaluator.results
print(f"Best model by k-mer correlation: {results['summary_statistics']['kmer_correlation_ranking'][0]}")
```

### Batch Evaluation
```python
# Evaluate multiple configurations
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]

for i, config in enumerate(configs):
    output_dir = f"evaluation_batch_{i}"
    evaluator = ModelEvaluator(config, output_dir)
    evaluator.run_evaluation()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_samples_per_batch` in config
   - Use `cuda_device: 0` to specify GPU

2. **Missing Model Files**
   - Check `base_path` in config
   - Verify checkpoint paths exist

3. **Oracle Model Download Issues**
   - Ensure `wandb` is properly configured
   - Check internet connection for artifact downloads

4. **Motif Analysis Fails**
   - Requires `grelu` with motif scanning capability
   - Check if JASPAR database is accessible

### Performance Tips

- **Faster Evaluation**: Reduce `num_sample_batches` for quick tests
- **Memory Efficient**: Process one model at a time by commenting out others in config
- **Parallel Processing**: Run multiple evaluations on different GPUs

## Extension Points

The evaluation system is designed to be extensible:

1. **Add New Metrics**: Extend `ModelEvaluator` class
2. **Custom Sampling**: Add new sampling methods to `generate_samples()`
3. **Additional Models**: Add model loading logic to `load_models()`
4. **Output Formats**: Modify `save_results()` for custom formats

## Notes

- This script reproduces all functionality from the original `eval.ipynb` notebook
- Results are compatible with the original analysis pipeline
- All random seeds are controlled for reproducible results
- Comprehensive logging helps with debugging and monitoring progress

For questions or issues, refer to the log files or contact the development team. 