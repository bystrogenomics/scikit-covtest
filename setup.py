from setuptools import setup, find_packages
setup(
    name='scikitcovtest',
    version='0.1.0',
    description='Covariance Hypothesis Testing and Evaluation',
    author='Austin Talbot',
    author_email='austin.talbot1993@gmail.com',
    url='https://github.com/bystrogenomics/scikit-covtest',
    packages=find_packages(),
    install_requires=[
        'numba',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'matplotlib',
        'pytest',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8',
)

