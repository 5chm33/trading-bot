from setuptools import setup, find_packages
from pathlib import Path

def get_long_description():
    here = Path(__file__).parent.resolve()
    readme = here / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""

def get_version(rel_path):
    here = Path(__file__).parent.resolve()
    for line in (here / rel_path).read_text(encoding="utf-8").splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Could not find version string.")

setup(
    name="trading_bot",
    version=get_version("src/__init__.py"),
    description="Algorithmic Trading Bot with RL",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Thomas Nance",
    author_email="thomasnance290@yahoo.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="algorithmic-trading reinforcement-learning quant",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "alpaca-py>=0.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "gymnasium>=0.28.1",
        "stable-baselines3>=1.8.0",
        "prometheus-client>=0.14.1",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "mypy>=0.910",
            "flake8>=3.9",
            "pytest-cov>=2.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "pyfolio>=0.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "tradebot-run=trading_bot.pipeline.paper_trading.executor:main",
            "tradebot-train=trading_bot.pipeline.training.trainer:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/5chm33/trading_bot/issues",
        "Source": "https://github.com/5chm33/trading_bot",
    },
    include_package_data=True,
)