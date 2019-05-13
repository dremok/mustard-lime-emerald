import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mle",
    version="0.0.5",
    author="dremok",
    author_email="max.y.leander@gmail.com",
    description="Utilities for common DS/ML/AI tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dremok/mustard-light-emerald",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas',
        'scikit-learn',
    ],
)
