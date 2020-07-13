import setuptools

from time_series_buffer import __version__ as tsb_version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time_series_buffer",
    version=tsb_version,
    author="Maximilian Gruber, Björn Ludwig",
    author_email="maximilian.gruber@ptb.de",
    description="This package provides support for buffering time-series with uncertainty.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PTB-PSt1/time-series-buffer",
    packages=setuptools.find_packages(),
    keywords="buffer time-series uncertainty metrology",
    install_requires=[
        "numpy",
        "uncertainties",
    ],
    classifiers=[
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
