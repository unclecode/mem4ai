import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the project version from package __init__.py
def read_version():
    with open(os.path.join('mem4ai', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    raise RuntimeError('Unable to find version string.')

setup(
    name='mem4ai',
    version=read_version(),
    author='unclecode',
    author_email='unclecode@kidocode.com',
    description='A powerful memory management library for LLMs and AI systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/unclecode/mem4ai',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=2.0.1',
        'scikit-learn>=1.5.1',
        'lmdb>=1.5.1',
        'litellm>=1.43.4',  # Adjust version as needed
        # Add any other core dependencies your project needs
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9',
            'black>=21.5b1',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx_rtd_theme>=0.5',
        ],
    },
    include_package_data=True,
    keywords='memory management llm ai',
    project_urls={
        'Bug Reports': 'https://github.com/unclecode/mem4ai/issues',
        'Source': 'https://github.com/unclecode/mem4ai/',
        'Documentation': 'https://mem4ai.readthedocs.io/',
    },
)