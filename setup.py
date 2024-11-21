from setuptools import setup, find_namespace_packages

VERSION="0.1.9"
DESCRIPTION = "meanderpy: a simple model of meandering"
with open("README.md", "r") as f:
	long_description_readme = f.read()

setup(
    name="meanderpy",
    version=VERSION,
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description=DESCRIPTION,
    keywords = 'rivers, meandering, geomorphology, stratigraphy',
    long_description=long_description_readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/meanderpy",
    download_url="https://github.com/zsylvester/meanderpy/archive/refs/tags/v{0}tar.gz".format(VERSION),
    packages=find_namespace_packages(include=['meanderpy', 'meanderpy.*']),
    install_requires=['numpy','matplotlib', 'scipy','numba','pillow','scikit-image','tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
		"Topic :: Scientific/Engineering :: Visualization",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
