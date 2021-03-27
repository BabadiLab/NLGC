from setuptools import setup, find_packages

setup(
    name='nlgc',
    version='0.3',
    description='Latent Causal inference for M/EEG data',
    author='Behrad Soleimani',
    packages=find_packages(),
    python_requires='>=3.8',

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'numba',
        'mne',
    ],

    # metadata for upload to PyPI
    author_email=['behrad@umd.edu', 'proloy@umd.edu'],
    license='BSD 3-Clause',
    project_urls={
        "Sorce Code": "https://github.com/proloyd/NLGC"
    }
)
