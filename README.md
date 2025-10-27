## Project summary
A reproducible MLOps demo that detects anomalies in stock time series using a univariate PyTorch autoencoder conditioned by sector. 
The objective is to provide timely risk signals to traders and asset managers for stocks within a selected sector.

The pipeline will cover:
- data ingestion (Yahoo Finance),
- preprocessing (adjusted close, scaling, sliding windows),
- model training with MLflow tracking,
- and a FastAPI service. 

Infrastructure will include Docker + docker‑compose, CI with GitHub Actions, and monitoring with Prometheus + Grafana. 
