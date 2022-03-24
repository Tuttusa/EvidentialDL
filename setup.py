from setuptools import setup

setup(name='evidentialdl',
      version='0.2',
      description='Evidential Deep learning implementation Pytorch',
      url='https://github.com/Tuttusa/EvidentialDL',
      author='Vaunorage',
      author_email='vaunorage@tuttusa.io',
      license='MIT',
      packages=['evidentialdl'],
      install_requires=[
          'torch==1.11.0',
          'numpy==1.21.5'
      ],
      zip_safe=False)
