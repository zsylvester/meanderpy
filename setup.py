import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meanderpy",
    version="0.1.1rc",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="A simple model of meander migration",
    keywords = 'rivers, meandering, geomorphology, stratigraphy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/meanderpy",
    packages=['meanderpy'],
    install_requires=['numpy','matplotlib','seaborn',
        'scipy','ipywidgets','numba','ipython'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
