import setuptools
import os


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("sparse_numeric_table", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="sparse_numeric_table_sebastian-achim-mueller",
    version=version,
    description="Read, write, and query sparse tables",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/sparse_numeric_table",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "sparse_numeric_table",
    ],
    package_data={"sparse_numeric_table": []},
    install_requires=[
        "numpy",
        "pandas",
        "dynamicsizerecarray>=0.0.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
