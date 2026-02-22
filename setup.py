#!/usr/bin/env python3
"""
Ensemble Trading System - Setup Script
"""

from setuptools import setup, find_packages

setup(
    name='ensemble-trading-system',
    version='1.0.0',
    description='Ensemble ML Trading System with XGBoost and LightGBM',
    author='Trading System Team',
    python_requires='>=3.9',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.1.0',
        'lightgbm>=3.3.0',
        'xgboost>=1.7.0',
        'optuna>=3.0.0',
        'matplotlib>=3.5.0',
        'joblib>=1.2.0',
        'tqdm>=4.64.0',
    ],
    entry_points={
        'console_scripts': [
            'trading-strategy=strategy:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
