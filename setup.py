from setuptools import find_packages, setup

setup(
    name="lab02",
    version="0.1.0",
    description="Refactored California housing machine learning lab.",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
