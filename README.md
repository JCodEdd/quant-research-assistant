# Quant Research Assistant (QRA)

A modular Python platform designed to support the development and testing of quantitative trading strategies.

## Vision
The goal of this project is to build an internal research platform that allows trading strategies to be defined, tested, and analyzed in a reproducible, professional, and modular way. It prioritizes high code quality, robust data handling, and anti-bias principles (e.g., prevention of lookahead bias) over "black box" automated trading.

## Current Stage: Module 1 (Data Ingestion)
The project is currently in its initial development phase, focusing on a robust data ingestion layer.
- **Completed**:
    - Data Ingestion Module (fetching, validating, and caching).
    - Monthly deterministic parquet partitioning for scalable storage.
    - Behavioral testing suite (AAA pattern) with 100% pass rate.
    - Standardized data validation (ensuring OHLCV integrity).
- **Next Phase (Upcoming)**:
    - Module 2: Feature Engineering (indicators calculation pipeline).

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment (see `.env.example`).
3. Run the ingestion pipeline:
   ```bash
   PYTHONPATH=src python main.py
   ```
4. Execute tests:
   ```bash
   PYTHONPATH=src pytest
   ```

