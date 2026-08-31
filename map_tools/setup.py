from setuptools import setup, find_packages

setup(
    name='map_tools',
    version='0.1.0',
    description='Ground Truth Map + Behavior Layer for CyberDog Blind Navigation PoC',
    author='RKH',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'open3d>=0.17.0',
        'Pillow>=9.5.0',
        'PyYAML>=6.0',
    ],
    python_requires='>=3.8',
)
