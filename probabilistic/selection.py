"""
Module for selecting the best hypothesis.
"""
import torch

class HypothesisSelector:
    def __init__(self, refinement_module=None):
        self.refinement_module = refinement_module

    def select(self, hypotheses, scores):
        """
        Selects the best hypothesis based on scores.
        Optionally refines the best hypothesis.
        
        Args:
            hypotheses (Tensor): A set of pose hypotheses.
            scores (Tensor): A tensor of scores for each hypothesis.
        
        Returns:
            The best pose hypothesis.
        """
        best_hypothesis_idx = torch.argmax(scores)
        best_hypothesis = hypotheses[best_hypothesis_idx]

        if self.refinement_module:
            best_hypothesis = self.refinement_module.refine(best_hypothesis)
            
        return best_hypothesis
