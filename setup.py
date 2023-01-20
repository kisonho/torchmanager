from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager-nightly',
    version="v1.1b5",
    description="PyTorch Training Manager v1.1 (Beta 5)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kison Ho',
    author_email='unfit-gothic.0q@icloud.com',
    packages=[
        'torchmanager',
        'torchmanager.callbacks',
        'torchmanager.data',
        'torchmanager.losses',
        'torchmanager.metrics',
        'torchmanager.train',
        'torchmanager_core',
        'torchmanager_core.devices',
        'torchmanager_core.errors',
        'torchmanager_core.view',
    ],
    package_dir={
        'torchmanager': 'lib',
        'torchmanager.callbacks': 'lib/callbacks',
        'torchmanager.data': 'lib/data',
        'torchmanager.losses': 'lib/losses',
        'torchmanager.metrics': 'lib/metrics',
        'torchmanager.train': 'lib/train',
        'torchmanager_core': 'core',
        'torchmanager_core.devices': 'core/devices',
        'torchmanager_core.errors': 'core/errors',
        'torchmanager_core.view': 'core/view',
    },
    install_requires=[
        'torch',
        'tqdm',
    ],
    python_requires=">=3.8",
    url="https://github.com/kisonho/torchmanager.git"
)
