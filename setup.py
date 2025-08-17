"""
Setup configuration for Behavioral Boredom Index package
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
__version__ = "1.0.0"

setup(
    name="behavioral-boredom-index",
    version=__version__,
    author="Senior ML Engineering Team",
    author_email="engineering@bbi-analytics.com",
    description="Privacy-preserving employee engagement analytics using federated learning and advanced AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/behavioral-boredom-index",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/behavioral-boredom-index/issues",
        "Source": "https://github.com/yourusername/behavioral-boredom-index",
        "Documentation": "https://docs.behavioral-boredom-index.com",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Human Resources",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
        ],
        "dashboard": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "dash>=2.11.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "optuna>=3.2.0",
            "mlflow>=2.5.0",
        ],
        "all": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bbi=bbi.cli:main",
            "bbi-dashboard=bbi.dashboard:main",
            "bbi-api=bbi.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bbi": [
            "models/*.pkl",
            "config/*.yaml",
            "data/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "employee-engagement",
        "behavioral-analytics", 
        "federated-learning",
        "privacy-preserving",
        "machine-learning",
        "nlp",
        "hr-analytics",
        "workforce-analytics",
        "predictive-modeling",
        "ai",
        "deep-learning",
        "sentiment-analysis",
        "turnover-prediction",
    ],
)
