from distutils.core import setup

setup(name='torchmanager',
      version='0.9.0',
      description='PyTorch training manager',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=['torchmanager'],
      package_dir={
            'torchmanager': 'lib'
      },
      url="https://github.com/kisonho/torchmanager.git"
)
