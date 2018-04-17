from setuptools import setup, find_packages

setup(
    name='highway-env',
    version='1.0.dev0',
    description='An environment for simulated highway driving tasks',
    url='https://github.com/eleurent/highway-env',
    author='Edouard Leurent',
    author_email='eleurent@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='autonomous highway driving simulation environment reinforcement learning',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=['gym', 'numpy', 'pygame', 'jupyter', 'matplotlib', 'pandas', 'pytest-runner'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
    },
    entry_points={
        'console_scripts': [],
    },
)

