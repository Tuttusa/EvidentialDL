from setuptools import setup

setup(name='evidentialdl',
      version='0.1',
      description='Evidential Deep learning implementation Pytorch',
      url='https://github.com/Tuttusa/EvidentialDL',
      author='Vaunorage',
      author_email='vaunorage@tuttusa.io',
      license='MIT',
      packages=['evidentialdl'],
      install_requires=[
          'torch',
          'numpy'
      ],
      zip_safe=False)
