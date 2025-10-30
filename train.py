"""
Main training script for RAP extensions.
"""
import argparse
from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from utils.general_utils import fix_seed

from uaas.trainer import UAASTrainer
from probabilistic.trainer import ProbabilisticTrainer
from semantic.trainer import SemanticTrainer

def main():
    parser = config_parser()
    model_params = ModelParams(parser)
    optimization = OptimizationParams(parser)
    
    parser.add_argument("--trainer_type", type=str, default="uaas", 
                        choices=["uaas", "probabilistic", "semantic"],
                        help="Type of trainer to use for the experiment.")
    
    # Add any other new arguments here
    parser.add_argument("--num_semantic_classes", type=int, default=19, 
                        help="Number of semantic classes for semantic trainer.")

    args = get_combined_args(parser)
    model_params.extract(args)
    optimization.extract(args)
    fix_seed(args.seed)

    trainer_map = {
        "uaas": UAASTrainer,
        "probabilistic": ProbabilisticTrainer,
        "semantic": SemanticTrainer
    }

    trainer_class = trainer_map[args.trainer_type]
    trainer = trainer_class(args)
    trainer.train()

if __name__ == "__main__":
    main()
