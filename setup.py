from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager',
      version='1.0.1-b1',
      description='PyTorch training manager (v1.0.1 Beta1)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=['torchmanager', 'torchmanager-nightly'],
      package_dir={
            'torchmanager': 'lib',
            'torchmanager-nightly': 'nightly'
      },
      python_requires=">=3.8",
      url="https://github.com/kisonho/torchmanager.git"
)
