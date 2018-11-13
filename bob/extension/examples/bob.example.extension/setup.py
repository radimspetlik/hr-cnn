#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring administrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.



# Add here other bob packages that your module depend on
setup_packages = ['bob.extension', 'bob.blitz']
bob_packages = []

from setuptools import setup, dist
dist.Distribution(dict(setup_requires = setup_packages + bob_packages))

# import the Extension class and the build_ext function from bob.blitz
from bob.blitz.extension import Extension, build_ext

# load the requirements.txt for additional requirements
from bob.extension.utils import load_requirements, find_packages
build_requires = setup_packages + bob_packages + load_requirements()

# read version from "version.txt" file
version = open("version.txt").read().rstrip()


# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name = 'bob.example.extension',
    version = version,
    description = 'Example for using Bob inside a C++ extension of a buildout project',

    url = 'https://github.com/<YourInstitution>/<YourPackage>',
    license = 'GPLv3',
    author = '<YourName>',
    author_email='<YourEmail>',
    keywords='bob, extension',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description = open('README.rst').read(),

    # This line is required for any distutils based packaging.
    # It will find all package-data inside the 'bob' directory.
    packages = find_packages('bob'),
    include_package_data = True,

    # These lines define which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    setup_requires = build_requires,
    install_requires = build_requires,

    # In fact, we are defining two extensions here. In any case, you can define
    # as many extensions as you need. Each of them will be compiled
    # independently into a separate .so file.
    ext_modules = [

      # The first extension defines the version of this package and all C++-dependencies.
      Extension("bob.example.extension.version",
        # list of files compiled into this extension
        [
          "bob/example/extension/version.cpp",
        ],
        # additional parameters, see Extension documentation
        version = version,
        bob_packages = bob_packages,
      ),

      # The second extension contains the actual C++ code and the Python bindings
      Extension("bob.example.extension._library",
        # list of files compiled into this extension
        [
          # the pure C++ code
          "bob/example/extension/Function.cpp",
          # the Python bindings
          "bob/example/extension/main.cpp",
        ],
        # additional parameters, see Extension documentation
        version = version,
        bob_packages = bob_packages,
      ),
    ],

    # Important! We need to tell setuptools that we want the extension to be
    # compiled with our build_ext function!
    cmdclass = {
      'build_ext': build_ext,
    },

    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points = {

      # scripts should be declared using this entry:
      'console_scripts' : [
        'bob_example_extension_reverse.py = bob.example.extension.script.reverse:main',
      ],
    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
