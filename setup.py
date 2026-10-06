from setuptools import setup, find_packages

setup(
    name="more",
    version="0.1.0",
    description="MoRE: Mixture of Reward Experts for Language-Based Trajectory Prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0,<5",
        "accelerate>=0.28.0",
        "datasets>=3.0.0",
        "peft>=0.10.0",
        "nltk>=3.8.0",
        "sentencepiece>=0.1.99",
        "scikit-learn>=1.0.0",
        "numpy>=1.22.0",
        "tqdm>=4.60.0",
    ],
)
