"""setup.py: setuptools control"""

import os

from setuptools import setup

IPL_PACKAGE_NAME = 'ipl'
FILE_LOCATION = os.path.dirname(__file__)
README_LOCATION = os.path.join(FILE_LOCATION, 'README.rst')
REQUIREMENTS_LOCATION = os.path.join(FILE_LOCATION, 'requirements.txt')
MANIFEST_LOCATION = os.path.join(FILE_LOCATION, 'MANIFEST.in')

with open(README_LOCATION, encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open(REQUIREMENTS_LOCATION, encoding='utf-8') as requirements_file:
    requirements = requirements_file.readlines()

with open(MANIFEST_LOCATION, encoding='utf-8') as manifest_file:
    include_lines = (line.strip() for line in manifest_file.readlines()
                     if line.strip().startswith('include'))
    package_data_patterns = list(line.split(' ')[-1] for line in include_lines)

extras = {
    'excel': ['xlsxwriter']
}

setup(
    name="cmdline-image-processor",
    packages=[IPL_PACKAGE_NAME],
    package_data={IPL_PACKAGE_NAME: package_data_patterns},
    include_package_data=True,
    entry_points={
        "console_scripts": ['ipl = ipl.image_processor:main']
    },
    version="0.1.2",
    license='MIT',
    description="Python command line application which analyses images",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Fugol Alina, Taran Anatoly",
    author_email="webdevAlina@gmail.com",
    url="https://github.com/webdevAlina1107/ImageAnalysis",
    download_url='https://github.com/webdevAlina1107/ImageAnalysis/archive/0.1.2.tar.gz',
    keywords=['IMAGE PROCESSING', 'CMD', 'UTILITY', 'RASTER'],
    install_requires=requirements,
    extras_require=extras,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
