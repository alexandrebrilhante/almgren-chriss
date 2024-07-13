# almgren-chriss

[![Build](https://github.com/alexandrebrilhante/almgren-chriss/actions/workflows/python-package.yml/badge.svg)](https://github.com/alexandrebrilhante/almgren-chriss/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/acrl)
![GitHub](https://img.shields.io/github/license/alexandrebrilhante/almgren-chriss)

Deep reinforcement learning for optimal execution of portfolio transactions.

## Installation
```bash
pip install acrl
```

## Usage
```python
from acrl import AlmgrenChriss

AlmgrenChriss(liquidation_time=60, n_trades=60, risk_aversion=0, episodes=10000).run()
```