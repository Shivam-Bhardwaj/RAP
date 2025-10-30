# Technical Documentation: Mathematical Foundations

This document provides rigorous mathematical formulations for RAP-ID extensions at PhD dissertation level.

## Table of Contents

1. [Uncertainty-Aware Adversarial Synthesis](#uncertainty-aware-adversarial-synthesis)
2. [Multi-Hypothesis Probabilistic APR](#multi-hypothesis-probabilistic-apr)
3. [Semantic-Adversarial Scene Synthesis](#semantic-adversarial-scene-synthesis)
4. [Baseline RAP Formulation](#baseline-rap-formulation)
5. [Theoretical Analysis](#theoretical-analysis)

---

## Uncertainty-Aware Adversarial Synthesis

### Problem Formulation

Given a training dataset $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{p}_i)\}_{i=1}^{N}$ where $\mathbf{x}_i \in \mathbb{R}^{H \times W \times 3}$ are input images and $\mathbf{p}_i \in \mathbb{R}^6$ are 6-DoF camera poses (3D translation $\mathbf{t} \in \mathbb{R}^3$ and rotation $\boldsymbol{\omega} \in \mathbb{R}^3$ represented as axis-angle), we seek to learn a probabilistic model:

$$p(\mathbf{p} | \mathbf{x}, \boldsymbol{\theta}) = \mathcal{N}(\mathbf{p}; \boldsymbol{\mu}_\theta(\mathbf{x}), \boldsymbol{\Sigma}_\theta(\mathbf{x}))$$

where $\boldsymbol{\mu}_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^6$ and $\boldsymbol{\Sigma}_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{6 \times 6}$ are parameterized by neural network $\boldsymbol{\theta}$.

### Uncertainty Decomposition

Following Kendall & Gal (2017), we decompose predictive uncertainty into:

**Aleatoric Uncertainty (Data-Dependent):**
$$\sigma_{\text{ale}}^2(\mathbf{x}) = \mathbb{E}_{p(\mathbf{p}|\mathbf{x})}[\text{Var}(\mathbf{p} | \mathbf{x}, \mathbf{y})]$$

This captures irreducible uncertainty due to sensor noise, occlusions, and ambiguity inherent in the data.

**Epistemic Uncertainty (Model-Dependent):**
$$\sigma_{\text{epi}}^2(\mathbf{x}) = \text{Var}_{p(\boldsymbol{\theta}|\mathcal{D})}[\mathbb{E}_{p(\mathbf{p}|\mathbf{x},\boldsymbol{\theta})}[\mathbf{p}]]$$

This captures reducible uncertainty due to limited training data or model capacity.

**Total Predictive Uncertainty:**
$$\sigma_{\text{total}}^2(\mathbf{x}) = \sigma_{\text{ale}}^2(\mathbf{x}) + \sigma_{\text{epi}}^2(\mathbf{x})$$

This follows from the law of total variance: $\text{Var}(Y) = \mathbb{E}[\text{Var}(Y|X)] + \text{Var}(\mathbb{E}[Y|X])$.

### Model Architecture

**Feature Extraction:**
Let $f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^D$ be the RAPNet backbone feature extractor:

$$\mathbf{h} = f_\theta(\mathbf{x})$$

**Pose Prediction Head:**
$$\boldsymbol{\mu}(\mathbf{x}) = \text{MLP}_\mu(\mathbf{h}) \in \mathbb{R}^{12}$$

where the pose is represented as a flattened $3 \times 4$ transformation matrix $\mathbf{P} \in \text{SE}(3)$.

**Aleatoric Uncertainty Head:**
$$\log \boldsymbol{\sigma}_{\text{ale}}^2(\mathbf{x}) = \text{MLP}_\sigma(\mathbf{h}) \in \mathbb{R}^6$$

where $\text{MLP}_\sigma$ has architecture:
- Linear: $D \rightarrow 128$
- ReLU
- Linear: $128 \rightarrow 6$

We model uncertainty per degree of freedom: 3 for translation, 3 for rotation.

### Epistemic Uncertainty via Monte Carlo Dropout

We approximate the posterior $p(\boldsymbol{\theta} | \mathcal{D})$ using Monte Carlo Dropout (Gal & Ghahramani, 2016). During training, dropout is applied with probability $p_{\text{drop}} = 0.1$. During inference:

$$\hat{\mathbf{p}}^{(m)} \sim f_{\boldsymbol{\theta}^{(m)}}(\mathbf{x}), \quad \boldsymbol{\theta}^{(m)} \sim q(\boldsymbol{\theta})$$

where $q(\boldsymbol{\theta})$ is the approximate posterior induced by dropout, and $m = 1, \ldots, M$ are Monte Carlo samples.

**Epistemic Uncertainty Estimate:**
$$\sigma_{\text{epi}}^2(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} (\hat{\mathbf{p}}^{(m)} - \bar{\mathbf{p}})^2$$

where $\bar{\mathbf{p}} = \frac{1}{M}\sum_{m=1}^{M} \hat{\mathbf{p}}^{(m)}$ is the posterior mean.

**Theoretical Justification:** Under the variational inference framework, dropout approximates a Bernoulli distribution over weights, which can be interpreted as a variational approximation to a Gaussian process posterior (Gal & Ghahramani, 2016).

### Uncertainty-Aware Loss Function

**Heteroscedastic Regression Loss:**

For aleatoric uncertainty, we use the negative log-likelihood under the Gaussian assumption:

$$\mathcal{L}_{\text{ale}} = -\log p(\mathbf{p}_{\text{gt}} | \mathbf{x}, \boldsymbol{\theta}) = \frac{1}{2}\sum_{d=1}^{6} \left[\frac{(p_d - \mu_d)^2}{\sigma_{\text{ale},d}^2} + \log \sigma_{\text{ale},d}^2 + \log(2\pi)\right]$$

In practice, we use the numerically stable formulation:

$$\mathcal{L}_{\text{ale}} = \frac{1}{2}\sum_{d=1}^{6} \left[\exp(-\log \sigma_{\text{ale},d}^2) \cdot (p_d - \mu_d)^2 + \log \sigma_{\text{ale},d}^2\right]$$

The first term is precision-weighted squared error, and the second term is a regularization penalty preventing uncertainty from growing unbounded.

**Decomposition:** We separate translation and rotation uncertainties:

$$\mathcal{L}_{\text{ale}} = \mathcal{L}_{\text{ale}}^t + \mathcal{L}_{\text{ale}}^r$$

where:
- Translation: $\mathcal{L}_{\text{ale}}^t = \frac{1}{2}\sum_{d=1}^{3} \left[\exp(-\log \sigma_{\text{ale},d}^2) \cdot \|\mathbf{t}_d - \hat{\mathbf{t}}_d\|_2^2 + \log \sigma_{\text{ale},d}^2\right]$
- Rotation: $\mathcal{L}_{\text{ale}}^r = \frac{1}{2}\sum_{d=4}^{6} \left[\exp(-\log \sigma_{\text{ale},d}^2) \cdot \|\boldsymbol{\omega}_d - \hat{\boldsymbol{\omega}}_d\|_2^2 + \log \sigma_{\text{ale},d}^2\right]$

### Uncertainty-Weighted Adversarial Training

**Adversarial Objective:**

The discriminator $D_\phi: \mathbb{R}^F \rightarrow [0,1]$ distinguishes between real features $\mathbf{f}_{\text{real}}$ from ground-truth images and fake features $\mathbf{f}_{\text{fake}}$ from rendered images:

$$\min_\phi \mathcal{L}_D = \mathbb{E}_{\mathbf{f}_{\text{real}}}[\|D_\phi(\mathbf{f}_{\text{real}}) - 1\|^2] + \mathbb{E}_{\mathbf{f}_{\text{fake}}}[\|D_\phi(\mathbf{f}_{\text{fake}}) - 0\|^2]$$

**Uncertainty-Weighted Generator Loss:**

We weight the adversarial loss by inverse uncertainty to prioritize uncertain regions:

$$w(\mathbf{x}) = \exp\left(-\frac{1}{6}\sum_{d=1}^{6} \sigma_{\text{total},d}(\mathbf{x})\right)$$

$$\mathcal{L}_{\text{adv}} = \lambda_{\text{adv}} \cdot \mathbb{E}_{\mathbf{f}_{\text{fake}}} \left[w(\mathbf{x}) \cdot \|D_\phi(\mathbf{f}_{\text{fake}}) - 1\|^2\right]$$

where $\lambda_{\text{adv}}$ is the adversarial loss weight.

**Intuition:** Regions with high uncertainty receive higher adversarial signal, encouraging the generator to produce more realistic renderings in these challenging areas.

### Uncertainty-Guided View Synthesis

**View Selection Strategy:**

We sample novel viewpoints $\mathbf{v}$ from a proposal distribution and select those with high uncertainty:

$$\mathcal{V}_{\text{sample}} = \{\mathbf{v} : \bar{\sigma}_{\text{total}}(\mathbf{v}) > \tau_{\text{uncertainty}}\}$$

where $\bar{\sigma}_{\text{total}}(\mathbf{v}) = \frac{1}{6}\sum_{d=1}^{6} \sigma_{\text{total},d}(\mathbf{v})$ is the mean uncertainty, and $\tau_{\text{uncertainty}}$ is a threshold.

**Perturbation Strategy:**

Given base poses $\{\mathbf{p}_i\}_{i=1}^{N}$, we generate perturbed poses:

$$\mathbf{p}_i' = \mathbf{p}_i + \boldsymbol{\delta}_t + \text{Rot}(\boldsymbol{\delta}_r)$$

where:
- Translation perturbation: $\boldsymbol{\delta}_t \sim \mathcal{U}(-\Delta_t, \Delta_t)^3$
- Rotation perturbation: $\boldsymbol{\delta}_r \sim \mathcal{U}(-\Delta_r, \Delta_r)^3$ converted to rotation matrix via exponential map

**3DGS Rendering:**

For each selected pose $\mathbf{p}_i'$, we render a synthetic image:

$$\mathbf{x}_{\text{synth}} = \mathcal{R}_{\text{3DGS}}(\mathbf{p}_i', \mathcal{G})$$

where $\mathcal{G}$ is the 3D Gaussian Splatting representation and $\mathcal{R}_{\text{3DGS}}$ is the differentiable renderer.

### Complete Training Objective

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{ale}} + \lambda_2 \mathcal{L}_{\text{feature}} + \lambda_3 \mathcal{L}_{\text{RVS}} + \lambda_4 \mathcal{L}_{\text{adv}}$$

where:
- $\mathcal{L}_{\text{feature}}$: Feature consistency loss (VICReg, Triplet, or MSE)
- $\mathcal{L}_{\text{RVS}}$: Pose loss on Random View Synthesis samples
- $\{\lambda_i\}_{i=1}^{4}$: Loss weights (default: $[1.0, 1.0, 1.0, 0.7]$)

---

## Multi-Hypothesis Probabilistic APR

### Problem Motivation

Single-point pose regression fails in ambiguous scenes where multiple poses explain the observations equally well. We model this as a multimodal distribution:

$$p(\mathbf{p} | \mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x}) \mathcal{N}(\mathbf{p}; \boldsymbol{\mu}_k(\mathbf{x}), \boldsymbol{\Sigma}_k(\mathbf{x}))$$

where $K$ is the number of mixture components.

### Mixture Density Network Architecture

**MDN Output Head:**

The network outputs parameters for a Gaussian mixture:

$$\{\pi_k, \boldsymbol{\mu}_k, \log \boldsymbol{\sigma}_k\}_{k=1}^{K} = h_\psi(\mathbf{h})$$

where:
- Mixture weights: $\boldsymbol{\pi} = \text{softmax}(\text{MLP}_\pi(\mathbf{h})) \in \mathbb{R}^K$ (logits)
- Means: $\boldsymbol{\mu}_k = \text{MLP}_\mu(\mathbf{h}) \in \mathbb{R}^{K \times 6}$
- Log-standard deviations: $\log \boldsymbol{\sigma}_k = \text{MLP}_\sigma(\mathbf{h}) \in \mathbb{R}^{K \times 6}$

**Architecture:**
- Input: Feature vector $\mathbf{h} \in \mathbb{R}^D$
- Output dimension: $K \cdot (1 + 6 + 6) = 13K$ (for $K=5$: 65 parameters)

**Diagonal Covariance Assumption:**

For computational efficiency, we assume diagonal covariance:

$$\boldsymbol{\Sigma}_k = \text{diag}(\exp(\log \boldsymbol{\sigma}_k^2)) = \begin{bmatrix}
\exp(\log \sigma_{k,1}^2) & 0 & \cdots & 0 \\
0 & \exp(\log \sigma_{k,2}^2) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \exp(\log \sigma_{k,6}^2)
\end{bmatrix}$$

This reduces parameter count from $K \cdot (1 + 6 + 21) = 28K$ to $13K$ while maintaining reasonable expressiveness.

### Negative Log-Likelihood Loss

**Likelihood Function:**

$$p(\mathbf{p}_{\text{gt}} | \mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x}) \mathcal{N}(\mathbf{p}_{\text{gt}}; \boldsymbol{\mu}_k(\mathbf{x}), \boldsymbol{\Sigma}_k(\mathbf{x}))$$

**Negative Log-Likelihood:**

$$\mathcal{L}_{\text{MDN}} = -\log \sum_{k=1}^{K} \pi_k \prod_{d=1}^{6} \mathcal{N}(p_{\text{gt},d}; \mu_{k,d}, \sigma_{k,d}^2)$$

**Numerically Stable Computation:**

Using log-sum-exp trick:

$$\mathcal{L}_{\text{MDN}} = -\log \sum_{k=1}^{K} \exp\left(\log \pi_k + \sum_{d=1}^{6} \log \mathcal{N}(p_{\text{gt},d}; \mu_{k,d}, \sigma_{k,d}^2)\right)$$

$$= -\text{logsumexp}_k \left[\log \pi_k - \frac{1}{2}\sum_{d=1}^{6}\left(\frac{(p_{\text{gt},d} - \mu_{k,d})^2}{\sigma_{k,d}^2} + \log(2\pi \sigma_{k,d}^2)\right)\right]$$

where $\text{logsumexp}_k$ is the numerically stable log-sum-exp operation.

### Hypothesis Sampling and Selection

**Sampling Hypotheses:**

Given the mixture distribution, we sample $M$ hypotheses:

$$\mathbf{p}_k^{(m)} \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad k \sim \text{Categorical}(\boldsymbol{\pi})$$

for $m = 1, \ldots, M$.

**Rendering-Based Validation:**

For each hypothesis $\mathbf{p}_k^{(m)}$, we render an image:

$$\mathbf{x}_{\text{rendered}}^{(m)} = \mathcal{R}_{\text{3DGS}}(\mathbf{p}_k^{(m)}, \mathcal{G})$$

**Similarity Scoring:**

We compute similarity between rendered and observed images:

$$s^{(m)} = \text{SSIM}(\mathbf{x}_{\text{rendered}}^{(m)}, \mathbf{x}_{\text{obs}}) + \lambda_{\text{perceptual}} \cdot \mathcal{L}_{\text{LPIPS}}(\mathbf{x}_{\text{rendered}}^{(m)}, \mathbf{x}_{\text{obs}})$$

where:
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity (Zhang et al., 2018)

**Best Hypothesis Selection:**

$$\mathbf{p}^* = \arg\max_{\mathbf{p}_k^{(m)}} s^{(m)}$$

**Refinement (Optional):**

After selection, we can refine the best hypothesis using geometric verification:

$$\mathbf{p}^*_{\text{refined}} = \arg\min_{\mathbf{p}} \sum_{j=1}^{J} \|\pi(\mathbf{X}_j, \mathbf{p}) - \mathbf{x}_j\|^2$$

where $\{\mathbf{X}_j, \mathbf{x}_j\}_{j=1}^{J}$ are 2D-3D correspondences, and $\pi$ is the projection function.

### Theoretical Properties

**Expressiveness:** A mixture of $K$ Gaussians can approximate any smooth probability density function arbitrarily well (McLachlan & Peel, 2000).

**Multimodality:** The mixture model naturally captures ambiguous scenes where multiple poses explain observations.

**Computational Complexity:** Inference: $O(K \cdot D)$; Hypothesis validation: $O(M \cdot R)$ where $R$ is rendering cost.

---

## Semantic-Adversarial Scene Synthesis

### Semantic Segmentation Integration

**Semantic Map Representation:**

Given semantic segmentation model $S: \mathbb{R}^{H \times W \times 3} \rightarrow \{0, 1, \ldots, C-1\}^{H \times W}$:

$$\mathbf{s} = S(\mathbf{x})$$

where $C$ is the number of semantic classes.

**Semantic-Aware Feature Fusion:**

We optionally fuse semantic features with image features:

$$\mathbf{h}_{\text{sem}} = \text{Enc}_\text{sem}(\mathbf{s}) \in \mathbb{R}^{D_s}$$

$$\mathbf{h}_{\text{fused}} = \text{Concat}(\mathbf{h}, \mathbf{h}_{\text{sem}}) \in \mathbb{R}^{D + D_s}$$

### Semantic Scene Manipulation

**Modification Function:**

For semantic class $c$, we define modification function $\mathcal{M}_c$:

$$\mathcal{M}_c(\mathbf{x}, \mathbf{s}, \mathbf{m}) = \begin{cases}
\mathcal{T}_c(\mathbf{x}, \mathbf{m}) & \text{if } s_{i,j} = c \\
\mathbf{x}_{i,j} & \text{otherwise}
\end{cases}$$

where $\mathcal{T}_c$ is a transformation specific to class $c$:

- **Sky:** $\mathcal{T}_{\text{sky}}(\mathbf{x}, \mathbf{m}) = \alpha \cdot \mathbf{x} + (1-\alpha) \cdot \mathbf{c}_{\text{sky}}$ (color shift)
- **Building:** $\mathcal{T}_{\text{building}}(\mathbf{x}, \mathbf{m}) = \mathbf{x} \odot (1 - \mathbf{m}) + \mathbf{x}_{\text{occluded}} \odot \mathbf{m}$ (occlusion)
- **Road:** $\mathcal{T}_{\text{road}}(\mathbf{x}, \mathbf{m}) = \mathbf{x} + \beta \cdot \boldsymbol{\epsilon}$ (noise injection)

### Adversarial Hard Negative Mining

**Objective:**

Find synthetic scenes that maximize pose prediction error:

$$\mathbf{x}_{\text{hard}} = \arg\max_{\mathbf{x}_{\text{synth}} \in \mathcal{X}_{\text{synth}}} \mathcal{L}_{\text{pose}}(f_\theta(\mathbf{x}_{\text{synth}}), \mathbf{p}_{\text{gt}})$$

**Gradient-Based Optimization:**

We optimize semantic modifications via gradient ascent:

$$\mathbf{m}_t = \mathbf{m}_{t-1} + \eta \cdot \nabla_{\mathbf{m}} \mathcal{L}_{\text{pose}}(f_\theta(\mathcal{R}(\mathbf{p}_{\text{gt}}, \mathbf{s}, \mathbf{m})), \mathbf{p}_{\text{gt}})$$

subject to constraints $\mathbf{m} \in [0, 1]^{H \times W}$.

**Approximation:** In practice, we use discrete sampling and select top-$K$ hardest examples.

### Curriculum Learning Schedule

**Difficulty Function:**

$$d_t = d_0 + \alpha \cdot \text{sigmoid}\left(\frac{\text{Perf}(t) - \tau_{\text{perf}}}{\Delta_{\text{perf}}}\right) \cdot (d_{\max} - d_0)$$

where:
- $d_0$: Initial difficulty (default: 0.1)
- $d_{\max}$: Maximum difficulty (default: 1.0)
- $\alpha$: Increment rate (default: 0.1)
- $\text{Perf}(t)$: Current performance metric (e.g., success rate)
- $\tau_{\text{perf}}$: Performance threshold (default: 0.8)
- $\Delta_{\text{perf}}$: Performance range for sigmoid (default: 0.1)

**Difficulty Application:**

Modification magnitude scales with difficulty:

$$\mathbf{m}_{\text{applied}} = d_t \cdot \mathbf{m}$$

**Theoretical Justification:** Curriculum learning has been shown to improve convergence and generalization (Bengio et al., 2009).

---

## Baseline RAP Formulation

### Original RAP Loss Functions

**Pose Loss:**

$$\mathcal{L}_{\text{pose}} = s_x \cdot \|\mathbf{t}_{\text{gt}} - \hat{\mathbf{t}}\|_p + s_q \cdot \|\mathbf{R}_{\text{gt}} - \hat{\mathbf{R}}\|_p$$

where $p \in \{1, 2\}$ is the norm order, and $s_x, s_q$ are learnable or fixed weights.

**Feature Loss:**

Depending on configuration:
- **VICReg:** Variance-Invariance-Covariance regularization
- **Triplet Loss:** $\max(0, \text{margin} + d(\mathbf{f}_{\text{anchor}}, \mathbf{f}_{\text{pos}}) - d(\mathbf{f}_{\text{anchor}}, \mathbf{f}_{\text{neg}}))$
- **NT-Xent:** Normalized Temperature-scaled Cross Entropy

**Random View Synthesis Loss:**

$$\mathcal{L}_{\text{RVS}} = \|\mathbf{p}_{\text{RVS}} - \hat{\mathbf{p}}_{\text{RVS}}\|_p$$

where $\mathbf{p}_{\text{RVS}}$ are poses from perturbed viewpoints.

**Adversarial Loss (with Discriminator):**

$$\mathcal{L}_{\text{adv}} = \|D_\phi(\mathbf{f}_{\text{fake}}) - 1\|^2$$

---

## Theoretical Analysis

### Convergence Analysis

**Uncertainty Calibration:**

Under the assumption of well-calibrated uncertainty, we expect:

$$\mathbb{E}[|\mathbf{p} - \hat{\mathbf{p}}|^2 | \sigma_{\text{total}}^2] \approx \sigma_{\text{total}}^2$$

**Calibration Error:**

We measure Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

where $B_m$ are bins partitioning uncertainty space, $\text{acc}(B_m)$ is accuracy in bin $m$, and $\text{conf}(B_m)$ is mean confidence.

### Generalization Bounds

**PAC-Bayesian Analysis:**

For MDN with $K$ components and $D$ feature dimensions:

$$R(\hat{f}) \leq \hat{R}(\hat{f}) + O\left(\sqrt{\frac{\log(K \cdot D) + \log(1/\delta)}{N}}\right)$$

where $R$ is generalization error, $\hat{R}$ is empirical error, and $\delta$ is confidence parameter.

### Information-Theoretic Perspective

**Information Gain:**

Uncertainty-guided sampling maximizes information gain:

$$I(\mathbf{p}; \mathbf{x}_{\text{new}} | \mathcal{D}) = H(\mathbf{p} | \mathcal{D}) - H(\mathbf{p} | \mathbf{x}_{\text{new}}, \mathcal{D})$$

Selecting high-uncertainty regions maximizes expected information gain.

---

## Implementation Details

### Numerical Stability

**Log-Variance Clamping:**

$$\log \boldsymbol{\sigma}^2 \leftarrow \text{clamp}(\log \boldsymbol{\sigma}^2, -10, 10)$$

Prevents numerical underflow/overflow in exponential operations.

**Log-Sum-Exp Trick:**

For mixture likelihood:

$$\log \sum_{k=1}^{K} \exp(a_k) = \max_k(a_k) + \log \sum_{k=1}^{K} \exp(a_k - \max_k(a_k))$$

### Training Dynamics

**Uncertainty Initialization:**

Aleatoric uncertainty initialized to small values: $\log \boldsymbol{\sigma}^2 \leftarrow -3$.

**Learning Rate Scheduling:**

We use ReduceLROnPlateau scheduler with factor 0.95 and patience based on validation loss.

**Early Stopping:**

Early stopping based on validation pose accuracy with patience parameter.

---

## References

1. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? NeurIPS.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. ICML.
3. Bishop, C. M. (1994). Mixture density networks. Technical Report.
4. Zhang, R., et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric. CVPR.
5. Bengio, Y., et al. (2009). Curriculum learning. ICML.
6. McLachlan, G., & Peel, D. (2000). Finite mixture models. Wiley.

