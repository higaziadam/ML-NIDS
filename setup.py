from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-nids",
    version="0.1.0",
    author="Adam Higazi",
    author_email="higaziadam03@gmail.com",
    description="A Machine Learning-based Network Intrusion Detection System (NIDS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/higaziadam/ML-NIDS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.3",
        "pandas>=2.3",
        "scikit-learn>=1.7",
        "scipy>=1.16",
        "pyarrow>=18",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "xgboost>=2.1",
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.30,<1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3",
            "pytest-cov>=6.0",
            "black>=24.10",
            "flake8>=7.1",
            "pylint>=3.3",
        ],
        "notebooks": [
            "jupyter>=1.1",
            "notebook>=7.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-nids-train=src.train:main",
            "ml-nids-predict=src.predict:main",
        ],
    },
)
