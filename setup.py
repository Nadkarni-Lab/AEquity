#!/usr/bin/env python3
from setuptools import setup, find_packages
# import versioneer

with open('README.md') as f:
    readme = f.read()

# with open('LICENSE.md') as f:
#     license = f.read()

dependencies = []
with open('requirements.txt', 'r') as f:
    for line in f:
        dependencies.append(line.strip())

setup(
    name='aequity',
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    # description='DESC',
    long_description=readme,
    author='Faris F. Gulamali, Ashwin S. Sawant, Girish N. Nadkarni',
    url='https://github.mountsinai.org/MSCIC/AEquity',
    version=0.1, 
    # license=license,
    python_requires=">=3.7",
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=dependencies
)