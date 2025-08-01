#!/usr/bin/env python3
"""
Comprehensive model evaluation script for DRAKES DNA models.
Converts eval.ipynb functionality to a standalone script with proper logging and configuration.
"""

import os
import sys
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Local imports
import diffusion_gosai_update
import diffusion_gosai_cfg
import dataloader_gosai
import oracle
from utils import set_seed

class ModelEvaluator:
    """Comprehensive model evaluation pipeline."""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Directory to save all outputs
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize random seed
        set_seed(self.config.get('seed', 0), use_cuda=True)
        
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.get('cuda_device', '0'))
        
        # Initialize results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_path': config_path,
                'output_dir': str(output_dir),
                'seed': self.config.get('seed', 0)
            },
            'log_likelihoods': {},
            'pred_activities': {},
            'atac_predictions': {},
            'kmer_correlations': {},
            'motif_analysis': {},
            'summary_statistics': {}
        }
        
        self.logger.info(f"Initialized ModelEvaluator with output directory: {output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all models specified in configuration."""
        self.logger.info("Loading models...")
        
        base_path = self.config['base_path']
        models = {}
        
        # Reinitialize Hydra
        GlobalHydra.instance().clear()
        initialize(config_path="configs_gosai", job_name="load_model", version_base="1.1")
        cfg = compose(config_name="config_gosai.yaml")
        
        # Load main finetuned model
        if 'finetuned_model' in self.config['models']:
            ckpt_path = os.path.join(base_path,self.config['models']['finetuned_model'])
            cfg.eval.checkpoint_path = ckpt_path
            model = diffusion_gosai_update.Diffusion(cfg, eval=False).cuda()
            model.load_state_dict(torch.load(cfg.eval.checkpoint_path))
            model.eval()
            models['finetuned'] = model
            self.logger.info(f"Loaded finetuned model from {ckpt_path}")
        
        # Load pretrained model
        if 'pretrained_model' in self.config['models']:
            old_path = os.path.join(base_path, self.config['models']['pretrained_model'])
            old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=cfg)
            old_model.eval()
            models['pretrained'] = old_model
            self.logger.info(f"Loaded pretrained model from {old_path}")
        
        # Load zero alpha model
        if 'zero_alpha_model' in self.config['models']:
            zero_alpha_path = os.path.join(base_path, self.config['models']['zero_alpha_model'])
            zero_alpha_model = diffusion_gosai_update.Diffusion(cfg).cuda()
            zero_alpha_model.load_state_dict(torch.load(zero_alpha_path))
            zero_alpha_model.eval()
            models['zero_alpha'] = zero_alpha_model
            self.logger.info(f"Loaded zero alpha model from {zero_alpha_path}")
        
        # Load CFG model
        if 'cfg_model' in self.config['models']:
            cfg_cfg = compose(config_name="config_gosai.yaml")
            cfg_cfg.model.cls_free_guidance = True
            cfg_cfg.model.cls_free_weight = self.config.get('cfg_weight', 10)
            cfg_cfg.model.cls_free_prob = self.config.get('cfg_prob', 0.1)
            cfg_path = os.path.join(base_path, self.config['models']['cfg_model'])
            cfg_cfg.eval.checkpoint_path = cfg_path
            cfg_model = diffusion_gosai_cfg.Diffusion(cfg_cfg, eval=False).cuda()
            cfg_model.load_state_dict(torch.load(cfg_cfg.eval.checkpoint_path)['state_dict'])
            cfg_model.eval()
            models['cfg'] = cfg_model
            self.logger.info(f"Loaded CFG model from {cfg_path}")
        
        return models
    
    def generate_samples(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, Any]]:
        """Generate samples from all models."""
        self.logger.info("Generating samples from models...")
        
        num_batches = self.config.get('num_sample_batches', 10)
        batch_size = self.config.get('num_samples_per_batch', 64)
        
        samples_data = {}
        
        # Initialize reward model once for all controlled sampling methods
        reward_model = None
        if 'pretrained' in models:
            reward_model = oracle.get_gosai_oracle(mode='train')
            reward_model.eval()
        
        for model_name, model in models.items():
            if model_name == 'pretrained':
                # Generate samples using multiple methods for the pretrained model
                sampling_methods = {
                    'pretrained': lambda m: m._sample(eval_sp_size=batch_size),
                    'pretrained_tds': lambda m: m.controlled_sample_TDS(
                        reward_model=reward_model, 
                        alpha=self.config.get('tds_alpha', 0.5),
                        guidance_scale=self.config.get('tds_guidance_scale', 1000),
                        eval_sp_size=batch_size
                    ),
                    'pretrained_cg': lambda m: m.controlled_sample_CG(
                        reward_model=reward_model,
                        guidance_scale=self.config.get('cg_guidance_scale', 300000),
                        eval_sp_size=batch_size
                    ),
                    'pretrained_smc': lambda m: m.controlled_sample_SMC(
                        reward_model=reward_model,
                        alpha=self.config.get('smc_alpha', 0.5),
                        eval_sp_size=batch_size
                    )
                }
                
                for method_name, sampling_func in sampling_methods.items():
                    self.logger.info(f"Generating samples for {method_name}...")
                    
                    all_detokenized_samples = []
                    all_raw_samples = []
                    
                    for _ in tqdm(range(num_batches), desc=f"Sampling {method_name}"):
                        samples = sampling_func(model)
                        all_raw_samples.append(samples)
                        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
                        all_detokenized_samples.extend(detokenized_samples)
                    
                    all_raw_samples = torch.concat(all_raw_samples)
                    
                    samples_data[method_name] = {
                        'detokenized': all_detokenized_samples,
                        'raw': all_raw_samples
                    }
                    
                    self.logger.info(f"Generated {len(all_detokenized_samples)} samples for {method_name}")
            else:
                # Handle other models normally
                self.logger.info(f"Generating samples for {model_name}...")
                
                all_detokenized_samples = []
                all_raw_samples = []
                
                for _ in tqdm(range(num_batches), desc=f"Sampling {model_name}"):
                    if model_name == 'cfg':
                        samples = model._sample(eval_sp_size=batch_size, w=self.config.get('cfg_weight', 10))
                    else:
                        samples = model._sample(eval_sp_size=batch_size)
                    
                    all_raw_samples.append(samples)
                    detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
                    all_detokenized_samples.extend(detokenized_samples)
                
                all_raw_samples = torch.concat(all_raw_samples)
                
                samples_data[model_name] = {
                    'detokenized': all_detokenized_samples,
                    'raw': all_raw_samples
                }
                
                self.logger.info(f"Generated {len(all_detokenized_samples)} samples for {model_name}")
        
        return samples_data
    
    def calculate_log_likelihoods(self, samples_data: Dict[str, Dict[str, Any]], models: Dict[str, torch.nn.Module]):
        """Calculate log-likelihoods for all samples."""
        self.logger.info("Calculating log-likelihoods...")
        
        # Use pretrained model for likelihood calculation (matching eval.ipynb behavior)
        likelihood_model = models.get('pretrained', list(models.values())[0])
        
        for model_name, data in samples_data.items():
            self.logger.info(f"Calculating likelihood for {model_name} samples...")
            
            logl = likelihood_model.get_likelihood(
                data['raw'], 
                num_steps=self.config.get('likelihood_steps', 128), 
                n_samples=1
            )
            
            logl_values = logl.detach().cpu().numpy()
            median_logl = np.median(logl_values)
            
            self.results['log_likelihoods'][model_name] = {
                'values': logl_values.tolist(),
                'median': float(median_logl),
                'mean': float(np.mean(logl_values)),
                'std': float(np.std(logl_values))
            }
            
            self.logger.info(f"{model_name} - Median log-likelihood: {median_logl:.4f}")
    
    def calculate_predictions(self, samples_data: Dict[str, Dict[str, Any]]):
        """Calculate oracle predictions for all samples."""
        self.logger.info("Calculating oracle predictions...")
        
        # GOSAI predictions
        for model_name, data in samples_data.items():
            self.logger.info(f"Calculating GOSAI predictions for {model_name}...")
            
            preds = oracle.cal_gosai_pred_new(data['detokenized'], mode='eval')
            median_pred = np.median(preds[:, 0])
            
            self.results['pred_activities'][model_name] = {
                'predictions': preds.tolist(),
                'median_hepg2': float(median_pred),
                'mean_hepg2': float(np.mean(preds[:, 0])),
                'std_hepg2': float(np.std(preds[:, 0]))
            }
            
            self.logger.info(f"{model_name} - Median HepG2 prediction: {median_pred:.4f}")
        
        # ATAC predictions
        for model_name, data in samples_data.items():
            self.logger.info(f"Calculating ATAC predictions for {model_name}...")
            
            atac_preds = oracle.cal_atac_pred_new(data['detokenized'])
            accessibility_rate = (atac_preds[:, 1] > 0.5).sum() / len(atac_preds)
            
            self.results['atac_predictions'][model_name] = {
                'predictions': atac_preds.tolist(),
                'accessibility_rate': float(accessibility_rate),
                'mean_accessibility': float(np.mean(atac_preds[:, 1])),
                'std_accessibility': float(np.std(atac_preds[:, 1]))
            }
            
            self.logger.info(f"{model_name} - ATAC accessibility rate: {accessibility_rate:.4f}")
    
    def calculate_kmer_correlations(self, samples_data: Dict[str, Dict[str, Any]]):
        """Calculate 3-mer Pearson correlations."""
        self.logger.info("Calculating k-mer correlations...")
        
        # Get high expression k-mers
        highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, _, _, _ = oracle.cal_highexp_kmers(return_clss=True)
        
        for model_name, data in samples_data.items():
            self.logger.info(f"Calculating k-mer correlation for {model_name}...")
            
            generated_kmer = oracle.count_kmers(data['detokenized'])
            
            # Calculate correlation with high expression k-mers
            kmer_set = set(highexp_kmers_999.keys()) | set(generated_kmer.keys())
            counts = np.zeros((len(kmer_set), 2))
            
            for i, kmer in enumerate(kmer_set):
                if kmer in highexp_kmers_999:
                    counts[i][1] = highexp_kmers_999[kmer] * len(data['detokenized']) / n_highexp_kmers_999
                if kmer in generated_kmer:
                    counts[i][0] = generated_kmer[kmer]
            
            correlation, p_value = pearsonr(counts[:, 0], counts[:, 1])
            
            self.results['kmer_correlations'][model_name] = {
                'pearson_correlation': float(correlation),
                'p_value': float(p_value),
                'n_kmers': int(len(kmer_set))
            }
            
            self.logger.info(f"{model_name} - K-mer Pearson correlation: {correlation:.4f} (p={p_value:.4e})")
    
    def motif_analysis(self, samples_data: Dict[str, Dict[str, Any]]):
        """Perform JASPAR motif analysis."""
        self.logger.info("Performing JASPAR motif analysis...")
        
        try:
            from grelu.interpret.motifs import scan_sequences
            
            # Get high expression sequences for comparison
            _, _, _, _, _, _, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)
            
            # Scan high expression sequences
            self.logger.info("Scanning high expression sequences...")
            motif_count_top = scan_sequences(highexp_seqs_999, 'jaspar')
            motif_count_top_sum = motif_count_top['motif'].value_counts()
            
            motif_summaries = {'top_data': motif_count_top_sum}
            
            for model_name, data in samples_data.items():
                self.logger.info(f"Scanning motifs for {model_name}...")
                
                motif_count = scan_sequences(data['detokenized'], 'jaspar')
                motif_count_sum = motif_count['motif'].value_counts()
                motif_summaries[model_name] = motif_count_sum
            
            # Create comparison dataframe
            motifs_summary = pd.concat(list(motif_summaries.values()), axis=1)
            motifs_summary.columns = list(motif_summaries.keys())
            motifs_summary = motifs_summary.fillna(0)
            
            # Calculate correlations
            correlations = motifs_summary.corr(method='spearman')
            self.results['motif_analysis'] = {
                'motif_counts': {name: counts.to_dict() for name, counts in motif_summaries.items()},
                'correlations': correlations.to_dict(),
                'top_correlations': correlations.loc['top_data'].to_dict(),
                'n_motifs_detected': int(len(motifs_summary))
            }
            
            # Save motif summary
            motifs_summary.to_csv(self.output_dir / 'motif_summary.csv')
            correlations.to_csv(self.output_dir / 'motif_correlations.csv')
            
            self.logger.info(f"Motif analysis completed.")
            
        except Exception as e:
            self.logger.error(f"Motif analysis failed: {e}")
            self.results['motif_analysis'] = {'error': str(e)}
    
    def generate_summary_statistics(self):
        """Generate summary statistics across all evaluations."""
        self.logger.info("Generating summary statistics...")
        
        summary = {}
        
        # Log-likelihood summary
        if self.results['log_likelihoods']:
            summary['log_likelihood_ranking'] = sorted(
                [(name, data['median']) for name, data in self.results['log_likelihoods'].items()],
                key=lambda x: x[1], reverse=True
            )
        
        # Prediction activity ranking
        if self.results['pred_activities']:
            summary['prediction_ranking'] = sorted(
                [(name, data['median_hepg2']) for name, data in self.results['pred_activities'].items()],
                key=lambda x: x[1], reverse=True
            )
        
        # ATAC accessibility ranking
        if self.results['atac_predictions']:
            summary['atac_ranking'] = sorted(
                [(name, data['accessibility_rate']) for name, data in self.results['atac_predictions'].items()],
                key=lambda x: x[1], reverse=True
            )
        
        # K-mer correlation ranking
        if self.results['kmer_correlations']:
            summary['kmer_correlation_ranking'] = sorted(
                [(name, data['pearson_correlation']) for name, data in self.results['kmer_correlations'].items()],
                key=lambda x: x[1], reverse=True
            )
        
        self.results['summary_statistics'] = summary
        
        # Log summary
        for metric, ranking in summary.items():
            self.logger.info(f"{metric}: {ranking}")
    
    def save_results(self):
        """Save all results to files."""
        self.logger.info("Saving results...")
        
        # Save complete results as JSON
        results_file = self.output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save results as pickle for Python objects
        pickle_file = self.output_dir / 'evaluation_results.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save samples data if available
        if hasattr(self, 'samples_data') and self.samples_data:
            self.logger.info("Saving samples data...")
            
            # Save detokenized samples as JSON
            detokenized_samples = {}
            for model_name, data in self.samples_data.items():
                detokenized_samples[model_name] = data['detokenized']
            
            samples_json_file = self.output_dir / 'samples_detokenized.json'
            with open(samples_json_file, 'w') as f:
                json.dump(detokenized_samples, f, indent=2)
            
            # Save complete samples data (including raw tensors) as pickle
            samples_pickle_file = self.output_dir / 'samples_data.pkl'
            with open(samples_pickle_file, 'wb') as f:
                pickle.dump(self.samples_data, f)
            
            self.logger.info(f"  - Detokenized samples: {samples_json_file}")
            self.logger.info(f"  - Complete samples data: {samples_pickle_file}")
        
        # Save summary as CSV
        summary_data = []
        for model_name in self.results.get('log_likelihoods', {}).keys():
            row = {'model': model_name}
            
            if model_name in self.results.get('log_likelihoods', {}):
                row['median_log_likelihood'] = self.results['log_likelihoods'][model_name]['median']
            
            if model_name in self.results.get('pred_activities', {}):
                row['median_hepg2_prediction'] = self.results['pred_activities'][model_name]['median_hepg2']
            
            if model_name in self.results.get('atac_predictions', {}):
                row['atac_accessibility_rate'] = self.results['atac_predictions'][model_name]['accessibility_rate']
            
            if model_name in self.results.get('kmer_correlations', {}):
                row['kmer_pearson_correlation'] = self.results['kmer_correlations'][model_name]['pearson_correlation']
            
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / 'evaluation_summary.csv', index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"  - Complete results: {results_file}")
        self.logger.info(f"  - Python objects: {pickle_file}")
        self.logger.info(f"  - Summary table: evaluation_summary.csv")
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting comprehensive model evaluation...")
        
        try:
            # Load models
            models = self.load_models()
            
            # Generate samples
            samples_data = self.generate_samples(models)
            
            # Store samples data for saving
            self.samples_data = samples_data
            
            # Calculate log-likelihoods
            self.calculate_log_likelihoods(samples_data, models)
            
            # Calculate predictions
            self.calculate_predictions(samples_data)
            
            # Calculate k-mer correlations
            self.calculate_kmer_correlations(samples_data)
            
            # Motif analysis
            self.motif_analysis(samples_data)
            
            # Generate summary
            self.generate_summary_statistics()
            
            # Save results
            self.save_results()
            
            self.logger.info("Evaluation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive DRAKES DNA model evaluation")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", help="Output directory for results", 
                       default="data_and_model/evaluation_results")
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = ModelEvaluator(args.config, args.output_dir)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main() 