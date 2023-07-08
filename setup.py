import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="acrl",
    version="0.0.3",
    author="Alexandre Brilhante",
    author_email="alexandre.brilhante@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brilhana/almgren-chriss",
    project_urls={
        "Bug Tracker": "https://github.com/brilhana/almgren-chriss/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
