import conda.cli

packages = ['docopt', 'scipy', 'h5py', 'opencv', 'boost=1.65.1', 'bob.blitz']

for package in packages:
    conda.cli.main('conda', 'install',  '-y', package)
