import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='sparse_numeric_table',
    version='0.0.1',
    description='Read, write, and manipulate sparse tables',
    long_description=long_description,
    url='https://github.com/cherenkov-plenoscope',
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    license='mit',
    packages=[
        'sparse_numeric_table',
    ],
    install_requires=[
       'numpy',
       'pandas',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
)
