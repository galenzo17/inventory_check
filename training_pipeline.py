#!/usr/bin/env python3
"""
Advanced Training Pipeline for Medical YOLO
Includes hyperparameter optimization and experiment tracking
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import wandb
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from medical_yolo import MedicalYOLO, create_medical_yolo_variants

class MedicalYOLOTrainer:
    """Advanced trainer for Medical YOLO models"""
    
    def __init__(self, config_path: str, experiment_name: str = "medical_yolo"):
        self.config = self.load_config(config_path)
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking
        self.setup_experiment_tracking()
        
        # Loss function
        self.loss_fn = v8DetectionLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'map50': [],
            'map95': [],
            'lr': []
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb"""
        if self.config.get('use_wandb', True):
            wandb.init(
                project="medical-yolo",
                name=self.experiment_name,
                config=self.config
            )
    
    def build_model(self, model_variant: str = 'medical_medium') -> nn.Module:
        """Build Medical YOLO model"""
        models = create_medical_yolo_variants()
        model = models[model_variant]
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_path'):
            print(f"Loading pretrained weights from {self.config['pretrained_path']}")
            checkpoint = torch.load(self.config['pretrained_path'], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model.to(self.device)
    
    def setup_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, Any]:
        """Setup optimizer and scheduler"""
        # Separate parameters for different learning rates
        backbone_params = []
        neck_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'neck' in name:
                neck_params.append(param)
            else:
                head_params.append(param)
        
        # Different learning rates for different parts
        param_groups = [
            {'params': backbone_params, 'lr': self.config['lr'] * 0.1},  # Lower LR for backbone
            {'params': neck_params, 'lr': self.config['lr']},
            {'params': head_params, 'lr': self.config['lr'] * 2.0}  # Higher LR for head
        ]
        
        # Optimizer
        if self.config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                param_groups,
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        elif self.config['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                param_groups,
                lr=self.config['lr'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.0005)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Scheduler
        if self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2)
            )
        elif self.config['scheduler'] == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.get('milestones', [100, 150]),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        
        return optimizer, scheduler
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup train and validation data loaders"""
        # Use YOLO's built-in dataset handling for now
        # This would be replaced with custom medical dataset loaders
        
        train_dataset = YOLO(self.config['model_path']).train(
            data=self.config['data_path'],
            epochs=1,  # Just to get the dataset
            save=False,
            verbose=False
        )
        
        # Placeholder - implement custom dataset loaders
        train_loader = None  # Would be actual DataLoader
        val_loader = None    # Would be actual DataLoader
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, epoch: int) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # Log batch metrics
            if self.config.get('use_wandb'):
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        return total_loss / num_batches
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        
        # Placeholder validation metrics
        metrics = {
            'val_loss': 0.0,
            'map50': 0.0,
            'map95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        
        metrics['val_loss'] = total_loss / len(val_loader)
        
        # TODO: Implement actual mAP calculation
        # This would use YOLO's validation metrics
        
        return metrics
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Any, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': self.best_map,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config['save_dir']) / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if metrics.get('map50', 0) > self.best_map:
            self.best_map = metrics['map50']
            best_path = Path(self.config['save_dir']) / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with mAP@0.5: {self.best_map:.4f}")
    
    def train(self, model_variant: str = 'medical_medium'):
        """Main training loop"""
        print(f"Starting training with {model_variant} variant")
        
        # Build model
        model = self.build_model(model_variant)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer(model)
        
        # Setup data loaders (placeholder)
        train_loader, val_loader = self.setup_data_loaders()
        
        # Create save directory
        Path(self.config['save_dir']).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, epoch)
            
            # Update scheduler
            scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['map50'].append(val_metrics['map50'])
            self.training_history['map95'].append(val_metrics['map95'])
            self.training_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Log metrics
            if self.config.get('use_wandb'):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics
                })
            
            # Save checkpoint
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics)
            
            # Early stopping
            if self.check_early_stopping(val_metrics):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final training history
        self.save_training_history()
        
        print("Training completed!")
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check early stopping criteria"""
        if not self.config.get('early_stopping'):
            return False
        
        patience = self.config.get('early_stopping_patience', 20)
        min_delta = self.config.get('early_stopping_delta', 0.001)
        
        # Simple early stopping based on validation loss
        if len(self.training_history['val_loss']) < patience:
            return False
        
        recent_losses = self.training_history['val_loss'][-patience:]
        best_recent = min(recent_losses)
        current_loss = metrics['val_loss']
        
        return current_loss > (best_recent + min_delta)
    
    def save_training_history(self):
        """Save training history"""
        history_path = Path(self.config['save_dir']) / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.training_history['train_loss']))
        
        # Loss curves
        ax1.plot(epochs, self.training_history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.training_history['val_loss'], label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mAP curves
        ax2.plot(epochs, self.training_history['map50'], label='mAP@0.5')
        ax2.plot(epochs, self.training_history['map95'], label='mAP@0.5:0.95')
        ax2.set_title('mAP Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.training_history['lr'])
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Combined metrics
        ax4.plot(epochs, self.training_history['map50'], label='mAP@0.5')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs, self.training_history['train_loss'], 'r--', label='Train Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP@0.5', color='b')
        ax4_twin.set_ylabel('Loss', color='r')
        ax4.set_title('Training Progress')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config['save_dir']) / f"{self.experiment_name}_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, config_path: str, n_trials: int = 50):
        self.base_config = self.load_config(config_path)
        self.n_trials = n_trials
    
    def load_config(self, config_path: str) -> Dict:
        """Load base configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optimization objective function"""
        # Suggest hyperparameters
        config = self.base_config.copy()
        config.update({
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'SGD']),
            'scheduler': trial.suggest_categorical('scheduler', ['CosineAnnealingWarmRestarts', 'MultiStepLR']),
            'grad_clip': trial.suggest_float('grad_clip', 0.1, 10.0),
            'use_wandb': False  # Disable wandb for optimization
        })
        
        # Create trainer
        trainer = MedicalYOLOTrainer(config, f"trial_{trial.number}")
        
        # Train model (shortened for optimization)
        config['epochs'] = 10  # Short training for quick evaluation
        trainer.config = config
        
        try:
            trainer.train()
            # Return best validation mAP
            return trainer.best_map
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print("Optimization completed!")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        # Save results
        results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': self.n_trials
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        'model_path': 'yolov8n.pt',
        'data_path': './datasets/medical_inventory/dataset.yaml',
        'epochs': 300,
        'batch_size': 32,
        'lr': 0.001,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts',
        'weight_decay': 0.01,
        'grad_clip': 10.0,
        'save_dir': './runs/train',
        'use_wandb': True,
        'early_stopping': True,
        'early_stopping_patience': 20,
        'early_stopping_delta': 0.001,
        'T_0': 10,
        'T_mult': 2
    }

def main():
    """Example usage"""
    # Create default config
    config = create_default_config()
    
    # Save config
    Path('configs').mkdir(exist_ok=True)
    with open('configs/training_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Training pipeline ready!")
    print("Use: trainer = MedicalYOLOTrainer('configs/training_config.yaml')")
    print("Then: trainer.train('medical_medium')")

if __name__ == "__main__":
    main()