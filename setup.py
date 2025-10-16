from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'MIND'
LONG_DESCRIPTION = 'Python implementation of MIND model'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="MIND",
    version=VERSION,
    author="Hanwen Xing",
    author_email="<hanwen.xing@wrh.ox.ac.uk>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['pandas~=2.2.1',
                      'numpy~=1.26.4',
                      'matplotlib~=3.8.0',
                      'torch~=2.2.2',
                      'openTSNE~=1.0.2',
                      'requests~=2.32.5'],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'first package'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
