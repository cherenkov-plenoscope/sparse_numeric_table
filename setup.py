import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="sparse_numeric_table_sebastian-achim-mueller",
    version="0.0.6",
    description="Read, write, and query sparse tables",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cherenkov-plenoscope/sparse_numeric_table",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=["sparse_numeric_table",],
    install_requires=["numpy", "pandas",],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
