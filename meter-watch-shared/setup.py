from setuptools import setup, find_packages

setup(
    name="meter_watch_shared",
    version="0.1.0",
    description="Shared utilities for meter-watch projects",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "redis>=5.0.0",
        "python-dotenv>=1.0.0",
    ],
)