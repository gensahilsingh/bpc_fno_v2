from setuptools import setup, find_packages

setup(
    name="bpc_fno",
    version="0.1.0",
    description="Biophysical Cardiac Fourier Neural Operator",
    author="BPC-FNO Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
        "neuraloperator>=0.3.0",
        "myokit>=1.35.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "wfdb>=4.1.0",
        "h5py>=3.9.0",
        "pandas>=2.0.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "test": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
