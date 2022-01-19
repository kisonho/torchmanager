from distutils.core import setup

setup(name='torchmanager',
      version='0.9.0',
      description='PyTorch training manager',
      author='Kison Ho',
      author_email='RobertHe.KsHo@wayne.edu',
      packages=['torchmanager'],
      package_dir={
            'torchmanager': 'lib'
      }
)
