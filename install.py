import conda.cli

packages = ['']

for package in packages:
    conda.cli.main('conda', 'install',  '-y', package)
