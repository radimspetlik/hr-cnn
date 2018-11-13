#!../bin/python

"""
This script will generate a dependency graph of the given bob package(s) using the external tool ``dot``.
Packages can be either specified as a list of ``--packages`` or read from (several) ``--package-files``.
The latter option is mostly useful to generate a dot graph for all Bob packages.

The output is written to the given ``--output-file``, writing either the specified intermediate ``--dot-file``, or a temporary file.
When the ``--plot-external-dependencies`` is selected, also external (Python-)dependencies will be plotted as well, in red ellipses.
"""

from __future__ import print_function
import subprocess
import pkg_resources
import tempfile, os

import argparse

def main(command_line_options = None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument("--packages", '-p', nargs = '+', default = [], help = "Which packages do you want to have dependencies for?")
  parser.add_argument("--package-files", '-P', nargs = '+', default = [], help = "Read packages from the given files (usually requirement?.txt files of bob).")
  parser.add_argument("--dot-file", "-W", help = "If specified, the .dot file is written to the given file.")
  parser.add_argument("--output-file", "-w", default="dependencies.png", help = "Specify the (.png) file to write.")
  parser.add_argument("--limit-packages", '-l', nargs = '+', default = ['bob', 'facereclib', 'antispoofing'], help = "Limit packages read from --package-files to the given namespaces")
  parser.add_argument("--plot-external-dependencies", '-X', action='store_true', help = "Include external dependencies into the plot?")
  parser.add_argument("--rank-base-tools-same", '-R', action = 'store_true', help = "Set the rank of packages bob.extension, bob.core and bob.blitz at the same size")
  parser.add_argument("--vertical", '-V', action = 'store_true', help = "Display the dot graph in vertical direction")
  parser.add_argument("--verbose", '-v', action = 'store_true', help = "Print more information")

  args = parser.parse_args(command_line_options)
  args.limit_packages = tuple(args.limit_packages)


  # collect packages
  packages = args.packages[:]
  for package_file in args.package_files:
    for line in open(package_file):
      splits = line.rstrip().split()
      packages.extend([p for p in splits if p not in packages and p.startswith(args.limit_packages)])

  # generate dependencies
  dependencies = {}
  cpp_dependencies = {}
  has_parents = set()

  # function to add dependencies of packages recursively
  def _add_recursive(p):
    # check if package already parsed
    if p not in dependencies:
      if args.verbose:
        print("Checking %s" % p)
      deps = pkg_resources.require(p)
      dependencies[p] = [d.key for d in deps[1:]]
      if args.plot_external_dependencies and p.startswith(args.limit_packages):
        # also load the C++ dependencies, stored in the .version package
        import importlib
        lib = importlib.import_module(p)
        try:
          cpp_dependencies[p] = [dep for dep in lib.version.externals.keys() if not dep.startswith(args.limit_packages)]
        except AttributeError:
          cpp_dependencies[p] = []

      for d in dependencies[p]:
        has_parents.add(d)
        _add_recursive(d)

  for package in packages:
    _add_recursive(package)

  # prune dependencies
  pruned_dependencies = {}
  for package in dependencies:
    indirect_dependencies = set(d for i in [dependencies[dep] for dep in dependencies[package] if dep in dependencies] for d in i)
    pruned_dependencies[package] = [dep for dep in dependencies[package] if dep not in indirect_dependencies]

  # split all dependencies that are from bob (i.e., that belong to the --limit-packages) or not
  bob = set(package for package in pruned_dependencies if package.startswith(args.limit_packages))
  non_bob = set(dep for package in bob for dep in pruned_dependencies[package] if not dep.startswith(args.limit_packages))
  final = set(package for package in bob if package not in has_parents)

  # also prune the C++ dependencies
  if args.plot_external_dependencies:
    pruned_cpp_dependencies = {}
    for package in cpp_dependencies:
      indirect_dependencies = set(d for i in [cpp_dependencies[dep] for dep in dependencies[package] if dep in dependencies and dep in cpp_dependencies] for d in i)
      pruned_cpp_dependencies[package] = [dep for dep in cpp_dependencies[package] if dep not in indirect_dependencies]

    cpp = set(dep for package in bob for dep in pruned_cpp_dependencies[package])


  # function to return a name for the package that can serve as a dot variable
  def _n(p):
    return p.replace(".", "_").replace("-","_").replace("+","X")

  # write dependency graph
  dot_file = args.dot_file if args.dot_file is not None else tempfile.mkstemp(suffix='.dot')[1]
  with open(dot_file, 'w') as f:
    # open plot
    f.write("digraph Bob {\n")
    if not args.vertical:
      f.write("\trankdir=LR;\n")
    # write bob packages in squares
    for package in bob:
      f.write('\t%s [label="%s",shape=box,group=%s,style=filled,color=%s];\n' % (_n(package), package, "_".join(package.split('.')[:2]), 'green' if package in final else 'lightblue'))

    # write non-bob packages in squares
    if args.plot_external_dependencies:
      for package in non_bob:
        f.write('\t%s [label="%s",shape=box,color=red];\n' % (_n(package), package))
      for package in cpp:
        f.write('\t%s [label="%s",color=red];\n' % (_n(package), package))

    # write dependencies
    for package in bob:
      for dep in pruned_dependencies[package]:
        if dep in bob:
          f.write('\t%s -> %s [color=blue];\n' % (_n(package), _n(dep)))
        elif args.plot_external_dependencies:
          f.write('\t%s -> %s [color=red,style=dashed];\n' % (_n(package), _n(dep)))

      if args.plot_external_dependencies:
        for dep in pruned_cpp_dependencies[package]:
          f.write('\t%s -> %s [color=red,style=dashed];\n' % (_n(package), _n(dep)))

    # rank base tools at the same level
    if args.rank_base_tools_same:
      f.write('\t{rank=same; bob_extension bob_core bob_blitz}\n')

    f.write("}\n")

  if args.verbose and args.dot_file is not None:
    print("Wrote dot file %s" % dot_file)

  # call dot
  call = ["dot",  '-y', '-o', args.output_file, '-Tpng:cairo:gd', dot_file]
  if args.verbose:
    call[1:1] = ['-v']
    print("Calling dot: '%s'" % " ".join(call))
  subprocess.call(call)

  if args.verbose:
    print("\nWrote file %s" % args.output_file)

  # clean-up
  if args.dot_file is None:
    os.remove(dot_file)
