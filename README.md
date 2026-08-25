# Quant Research Assistant (QRA)

A modular Python platform designed to support the development and testing of quantitative trading strategies.

## Vision
The goal of this project is to build an internal research platform that allows trading strategies to be defined, tested, and analyzed in a reproducible, professional, and modular way. It prioritizes high code quality, robust data handling, and anti-bias principles (e.g., prevention of lookahead bias) over "black box" automated trading.

## Current Stage: Module 2 (Feature Engineering & Research Orchestration)
The project has evolved into a fully functional research pipeline.
- **Completed**:
    - **Module 1 (Data Ingestion)**: Robust fetching, validation, and caching.
    - **Module 2 (Feature Engineering)**: Calculation pipeline for technical indicators (SMA, EMA, RSI, ATR, Volatility, etc.).
    - **Orchestration**: `ResearchOrchestrator` implementation for unified data flow.
- **Next Phase (Upcoming)**:
    - Module 3: Strategy & Signal Generation.

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

## License
MIT
