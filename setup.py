"""setup.py: setuptools control"""

import os

from setuptools import setup

IPL_PACKAGE_NAME = 'ipl'
PACKAGE_VERSION = '0.2.1'
FILE_LOCATION = os.path.dirname(__file__)
README_LOCATION = os.path.join(FILE_LOCATION, 'README.rst')
REQUIREMENTS_LOCATION = os.path.join(FILE_LOCATION, 'requirements.txt')

with open(README_LOCATION, encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open(REQUIREMENTS_LOCATION, encoding='utf-8') as requirements_file:
    requirements = requirements_file.readlines()

extras = {
    'excel': ['xlsxwriter']
}

setup(
    name="cmdline-image-processor",
    packages=[IPL_PACKAGE_NAME],
    include_package_data=True,
    entry_points={
        "console_scripts": [f'{IPL_PACKAGE_NAME} = {IPL_PACKAGE_NAME}.entry:main']
    },
    version=PACKAGE_VERSION,
    license='MIT',
    description="Python command line application which analyses images data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Fugol Alina, Taran Anatoly",
    author_email="webdevAlina@gmail.com",
    url="https://github.com/webdevAlina1107/ImageAnalysis",
    download_url=f'https://github.com/webdevAlina1107/ImageAnalysis/archive/{PACKAGE_VERSION}.tar.gz',
    keywords=['IMAGE PROCESSING', 'CMD', 'UTILITY', 'RASTER'],
    install_requires=requirements,
    extras_require=extras,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
