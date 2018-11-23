import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meanderpy",
    version="1.0.0",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="A simple model of meander migration",
    keywords = 'rivers, meandering, geomorphology, stratigraphy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/meanderpy",
    packages=setuptools.find_packages(),
    install_requires=['numpy','matplotlib','seaborn',
        'scipy','ipywidgets','numba','ipython']
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
