---
toc: false
math: true
---



![coding](/assets/pytorch-icon.png)

## `Notes` Overview

[check notes](./docs)

- PyTorch is a popular deep learning framework known for its flexibility and dynamic computation graphs.
- These notes will cover essential PyTorch concepts, including tensors, autograd, modules, and optimizers.
- Example codes will be provided for common tasks such as model building, training loops, and data loading.
- Guidance on integrating PyTorch with PyTorch Lightning for streamlined training and experiment management.
- Tips and best practices for debugging, performance optimization, and reproducibility.
- Useful references and resources for further learning.

\[
\begin{aligned}
KL(\hat{y} || y) &= \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}} \\
JS(\hat{y} || y) &= \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))
\end{aligned}
\]
