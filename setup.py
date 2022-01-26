from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager',
      version='0.9.3',
      description='PyTorch training manager (v0.9 Beta4)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=['torchmanager'],
      package_dir={'torchmanager': 'lib'},
      python_requires=">=3.8",
      url="https://github.com/kisonho/torchmanager.git"
)
