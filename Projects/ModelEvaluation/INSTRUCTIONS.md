## Model Evaluation with Wandb

### 1. Register Dataset
Generate and register a dataset for a particular model use case

```python
python dataset_generator.py
```

### 2. Train Model

```python
python model_trainer.py
python model_trainer.py --validation_split 0.05
python model_trainer.py --batch_size 64
```

### 3. Evaluate Model

```python
python model_evaluator.py
```