# GLBA DistilBERT Configuration Files

This directory contains configuration files for all aspects of the GLBA DistilBERT project.

## Configuration Files

### 1. `model_config.yaml`
Main model and training configuration including:
- Model architecture parameters (DistilBERT settings, dropout, etc.)
- Training hyperparameters (learning rates, batch size, epochs)
- Data processing settings
- Logging and evaluation preferences
- GPU/device configuration

### 2. `data_collection_config.yaml`
Configuration for data collection from various sources:
- **Westlaw**: Legal database scraping settings
- **Nexis Uni**: Academic and legal content collection
- **SEC**: Securities and Exchange Commission filings
- **Federal Sources**: Federal Register, FTC documents
- Document processing and validation rules
- Rate limiting and error handling

### 3. `preprocessing_config.yaml`
Text preprocessing and cleaning configuration:
- Text cleaning rules for legal documents
- Financial term normalization
- Domain-specific tokenization
- Text segmentation strategies
- Quality filtering criteria
- Data augmentation (optional)

### 4. `labeling_config.yaml`
Automatic labeling configuration for GLBA violations:
- Keywords and patterns for each violation type
- Confidence thresholds for labeling
- Data balancing strategies
- Quality control settings
- Output file paths

### 5. `evaluation_config.yaml`
Comprehensive evaluation configuration:
- Metrics computation (standard ML + domain-specific)
- Visualization settings
- Error analysis configuration
- Model interpretability settings
- Reporting formats and templates

## Usage

These configuration files are designed to be used with YAML config loaders. Example usage:

```python
import yaml
from pathlib import Path

# Load model configuration
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in training
from src.model.trainer import GLBATrainer, create_trainer_config

trainer_config = create_trainer_config(**config['training'])
```

## Customization

You can customize these configurations for different experiments:

1. **Development**: Use smaller batch sizes, fewer epochs
2. **Production**: Enable GPU, increase batch sizes, use wandb logging
3. **Research**: Enable data augmentation, cross-validation
4. **Compliance**: Focus on high precision, enable interpretability

## Environment-Specific Configs

Consider creating environment-specific versions:
- `model_config_dev.yaml` - Development settings
- `model_config_prod.yaml` - Production settings
- `model_config_research.yaml` - Research experiments

## Configuration Validation

The system validates configurations at runtime. Key requirements:
- Required fields must be present
- Numeric values must be within valid ranges
- File paths must exist (for input files)
- GPU settings match available hardware

## Best Practices

1. **Version Control**: Keep configurations in version control
2. **Documentation**: Comment complex settings
3. **Validation**: Test configurations before long training runs
4. **Backup**: Save successful configurations with model checkpoints
5. **Parameterization**: Use environment variables for sensitive settings

## Configuration Hierarchy

Settings are applied in this order (later overrides earlier):
1. Default values in code
2. Base configuration file
3. Environment-specific overrides
4. Command-line arguments
5. Environment variables