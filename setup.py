"""setup.py: setuptools control"""

import os

from setuptools import find_packages, setup

directory_location = os.path.dirname(__file__)
readme_location = os.path.join(directory_location, 'README.md')
requirements_location = os.path.join(directory_location, 'requirements.txt')

with open(readme_location, "rb") as f:
    long_description = f.read().decode("utf-8")

with open(requirements_location, 'r') as requirements_file:
    requirements = requirements_file.read().split('\n')

setup(
    name="cmdline-image-processor",
    packages=find_packages(),
    entry_points={
        "console_scripts": ['ipl = ipl.image_processor:main']
    },
    version="0.0.1",
    license='MIT',
    description="Python command line application which analyses images",
    long_description=long_description,
    author="Fugol Alina, Taran Anatoly",
    author_email="webdevAlina@gmail.com",
    url="https://github.com/Mtrqq/LPPSolver/blob/master/lpp_solver",
    keywords=['IMAGE PROCESSING', 'CMD', 'UTILITY', 'RASTER'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
