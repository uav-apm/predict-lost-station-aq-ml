from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md")


setup(
    name="aq-spatial-reconstruction",
    version="0.1.0",
    description="Research scaffold for reconstructing and forecasting a lost air-quality station from neighboring stations",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    install_requires=[
        "pandas>=2.1",
        "numpy>=1.26",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "joblib>=1.3",
        "matplotlib>=3.8",
        "plotly>=5.24",
        "pytest>=8.0",
    ],
    extras_require={
        "lstm": [
            "tensorflow>=2.15,<3; python_version < '3.13'",
        ],
        "npu": [
            "tensorflow>=2.15,<3; python_version < '3.13'",
            "intel-extension-for-tensorflow>=2.15; python_version < '3.13'",
        ],
    },
    entry_points={
        "console_scripts": [
            "aq-spatial-reconstruction-train=src.cli_train:train_cmd",
            "aq-spatial-reconstruction-eval=src.cli_eval:eval_cmd",
            "aq-spatial-reconstruction-predict=src.cli_predict:predict_cmd",
            "aq-spatial-reconstruction-plot=src.cli_plot:plot_cmd",
        ]
    },
    include_package_data=True,
)
