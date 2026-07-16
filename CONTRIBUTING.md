# Contributing to ChurnRadar

Thank you for your interest in contributing to ChurnRadar! We welcome contributions to improve the machine learning pipeline, analytics dashboards, and user interface.

## How to Contribute

1. **Fork the Repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction-platform.git
   cd customer-churn-prediction-platform
   ```
3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```
5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Train the ML model** to generate pipeline files:
   ```bash
   python train_model.py
   ```
7. **Make your changes** and verify they work:
   ```bash
   python test_pipeline.py
   streamlit run Home.py
   ```
8. **Commit your changes** with descriptive messages:
   ```bash
   git commit -am "feat: add customer retention chart to dashboard"
   ```
9. **Push to your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```
10. **Open a Pull Request** against the main repository.

## Coding Guidelines

- **PEP 8 Compliance**: Follow Python PEP 8 styling conventions.
- **Type Hints**: Include type hints for function signatures.
- **Docstrings**: Document classes, methods, and functions using docstrings.
- **No Incomplete Code**: Do not commit commented-out placeholder code or unfinished tasks.
