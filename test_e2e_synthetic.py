#!/usr/bin/env python3
"""
End-to-End Test Runner for Synthetic Data

Runs comprehensive e2e tests on synthetic dataset without requiring GPU or full GS training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from uaas.trainer import UAASTrainer
from probabilistic.trainer import ProbabilisticTrainer
from semantic.trainer import SemanticTrainer
from arguments.options import config_parser
import tempfile
import torch

def test_trainer_initialization():
    """Test that all trainers can be initialized with synthetic dataset."""
    dataset_path = 'synthetic_test_dataset'
    model_path = 'synthetic_test_dataset/model'
    
    if not os.path.exists(dataset_path):
        print(f"✗ Synthetic dataset not found at {dataset_path}")
        print("  Run: python tests/synthetic_dataset.py --source data/Cambridge/KingsCollege/colmap --output synthetic_test_dataset --num_train 10 --num_test 5")
        return False
    
    parser = config_parser()
    base_args = ['--run_name', 'test_e2e', '--datadir', dataset_path]
    
    results = {}
    
    # Test UAAS Trainer
    print("\n" + "="*60)
    print("Testing UAAS Trainer")
    print("="*60)
    try:
        args = parser.parse_args(base_args)
        args.device = 'cpu'
        args.render_device = 'cpu'
        args.resolution = 1.0  # Add resolution
        args.logbase = tempfile.mkdtemp()
        args.model_path = model_path
        args.dataset_type = 'Colmap'
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        args.val_batch_size = 1
        args.compile = False
        
        trainer = UAASTrainer(args)
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'sampler')
        assert hasattr(trainer, 'adversarial_loss')
        print("✓ UAAS Trainer initialized successfully")
        results['UAAS'] = True
    except Exception as e:
        print(f"✗ UAAS Trainer failed: {e}")
        results['UAAS'] = False
    
    # Test Probabilistic Trainer
    print("\n" + "="*60)
    print("Testing Probabilistic Trainer")
    print("="*60)
    try:
        args = parser.parse_args(base_args)
        args.device = 'cpu'
        args.render_device = 'cpu'
        args.resolution = 1.0  # Add resolution
        args.logbase = tempfile.mkdtemp()
        args.model_path = model_path
        args.dataset_type = 'Colmap'
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        args.compile = False
        
        trainer = ProbabilisticTrainer(args)
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'hypothesis_validator')
        assert hasattr(trainer, 'criterion')
        print("✓ Probabilistic Trainer initialized successfully")
        results['Probabilistic'] = True
    except Exception as e:
        print(f"✗ Probabilistic Trainer failed: {e}")
        results['Probabilistic'] = False
    
    # Test Semantic Trainer
    print("\n" + "="*60)
    print("Testing Semantic Trainer")
    print("="*60)
    try:
        args = parser.parse_args(base_args)
        args.device = 'cpu'
        args.render_device = 'cpu'
        args.resolution = 1.0  # Add resolution
        args.logbase = tempfile.mkdtemp()
        args.model_path = model_path
        args.dataset_type = 'Colmap'
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        args.num_semantic_classes = 19
        args.compile = False
        
        trainer = SemanticTrainer(args)
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'synthesizer')
        assert hasattr(trainer, 'hard_negative_miner')
        assert hasattr(trainer, 'curriculum')
        print("✓ Semantic Trainer initialized successfully")
        results['Semantic'] = True
    except Exception as e:
        print(f"✗ Semantic Trainer failed: {e}")
        results['Semantic'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    return all_passed

if __name__ == "__main__":
    success = test_trainer_initialization()
    sys.exit(0 if success else 1)

