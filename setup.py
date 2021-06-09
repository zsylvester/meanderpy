import setuptools

long_description = """\
'meanderpy' is a Python module that implements a simple numerical model of meandering, the one described by Howard & Knutson in their 1984 paper "Sufficient Conditions for River Meandering: A Simulation Approach". This is a kinematic model that is based on computing migration rate as the weighted sum of upstream curvatures; flow velocity does not enter the equation. Curvature is transformed into a 'nominal migration rate' through multiplication with a migration rate (or erodibility) constant.
"""

setuptools.setup(
    name="meanderpy",
    version="0.1.8",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="meanderpy: a simple model of meandering",
    keywords = 'rivers, meandering, geomorphology, stratigraphy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/meanderpy",
    packages=['meanderpy'],
    # scripts=['/Users/zoltan/Dropbox/Channels/meanderpy/meanderpy/meanderpy.py'],
    install_requires=['numpy','matplotlib','seaborn',
        'scipy','numba','pillow','scikit-image','tqdm'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
