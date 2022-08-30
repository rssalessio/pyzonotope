from setuptools import setup, find_packages
from os import path


setup(name = 'pyzonotope',
    packages=find_packages(),
    version = '0.0.5',
    description = 'Zonotopes in Python',
    url = 'https://github.com/rssalessio/pyzonotope',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)