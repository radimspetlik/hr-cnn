#!/usr/bin/env python

"""
This script helps you to release a new version of a Bob package.
There are some considerations that needs to be taken into account **before**
you release a new version:

  * Make sure you running this script from the root directory of the package.
  * Make sure all the tests are passing.
  * Make sure the documentation is building with the following command:
    ``sphinx-build -aEWn doc sphinx``
  * Make sure the documentation badges in README.rst are pointing to:
    https://www.idiap.ch/software/bob/docs/bob/...
  * Make sure all changes are committed to the git repository and pushed.
  * For database packages, make sure that the '.sql3' file or other
    metadata files have been generated (if any).
  * Make sure bob.nightlies is green after the changes are submitted if the
    package is a part of the nightlies.
  * Make sure you follow semantic versions: http://semver.org
  * Make sure that the `stable` version that you trying to release is not
    already released.
  * If your package depends on an unreleased version of another Bob package,
    you need to release that package first.

Also, there are some manual steps to be taken into account **after** this
script completes successfully.

  * Please write down the changes that are applied to the package between last
    stable version and this version that you are trying to release. This
    changes are written to release tags of packages in the Gitlab interface.
    For an example look at: https://gitlab.idiap.ch/bob/bob.extension/tags
  * If you are at Idiap, once the CI for the latest tag finishes, it will
    *automatically* create a merge request for that package on bob.conda.
    Your job is to make sure that merge requests is green and merged too so
    that the conda package for this package is released to our channel.
    You will not get any emails for this merge request. You have to find it
    yourself on bob.conda after the CI finishes in the package.

The 'stable' version (i.e., what will be downloadable from PyPI) can be
current version of the package, but not lower than that.

The 'latest' version (i.e., what will be the new master branch on Gitlab)
must be higher than the current and than the stable version.

By default, both versions can be automatically computed from the 'current'
version, which is read from the 'version.txt' file. In this case, the
'stable' version will be the 'current' version without the trailing beta
indicator (unless --minor or --major is specified), and the 'latest' version
will be 1 patch level above the 'current' version, with the beta indicator 0,
for example:

* current version (in version.txt): 2.1.6b3
-> automatic stable version: 2.1.6
-> automatic latest version: 2.1.7b0

* current version (in version.txt): 2.1.6b3 and --minor is provided.
-> automatic stable version: 2.2.0
-> automatic latest version: 2.2.1b0

* current version (in version.txt): 2.1.6b3 and --major is provided.
-> automatic stable version: 3.0.0
-> automatic latest version: 3.0.1b0


By default, this script executes two steps, in this order:

  * tag: If given, the 'stable' version will be set and added to Gitlab;
    and the version is tagged in Gitlab and pushed.

  * latest: The 'latest' version will be set and committed to Gitlab

If any of these commands fail, the remaining steps will be skipped, unless you
specify the '--keep-going' option.

If you only want a subset of the steps to be executed, you can limit them using
the '--steps' option. A valid use case, e.g., is only to re-upload the
documentation.

Examples:

  Tags my package with the stable version '2.0.0'. Update my next package
  version to '2.0.1a0'. Do it verbosely ('-vv'):

    %(prog)s --stable-version=2.0.0 --latest-version=2.0.1a0 -vv


  Print out, what would be done using the '--dry-run' option:

    %(prog)s -q


  Do everything automatically (assumes a proper version.txt file):

    %(prog)s -vv
"""

from __future__ import print_function
import sys
import os
import subprocess
import logging
import re

import argparse
from distutils.version import StrictVersion as Version

logger = logging.getLogger("bob.extension")


def _update_readme(version=None):
  # replace the travis badge in the README.rst with the given version
  DOC_IMAGE = re.compile(r'\-(stable|(v\d+\.\d+\.\d+([abc]\d+)?))\-')
  BRANCH_RE = re.compile(r'/(stable|master|(v\d+\.\d+\.\d+([abc]\d+)?))')
  with open("README.rst") as read:
    with open(".README.rst", 'w') as write:
      for line in read:
        if BRANCH_RE.search(line) is not None:
          if "gitlab" in line: #gitlab links
            replacement = "/v%s" % version if version is not None else "/master"
            line = BRANCH_RE.sub(replacement, line)
          if "software/bob" in line: #our doc server
            if 'master' not in line: #don't replace 'latest' pointer
              replacement = "/v%s" % version if version is not None \
                  else "/stable"
              line = BRANCH_RE.sub(replacement, line)
        if DOC_IMAGE.search(line) is not None:
          replacement = '-v%s-' % version if version is not None else '-stable-'
          line = DOC_IMAGE.sub(replacement, line)
        write.write(line)
  os.rename(".README.rst", "README.rst")


def main(command_line_options=None):
  doc = __doc__ % dict(prog=os.path.basename(sys.argv[0]))
  parser = argparse.ArgumentParser(
      description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument("--latest-version", '-l',
                      help="The latest version for the package; if not specified, it is guessed from the current version")
  parser.add_argument("--stable-version", '-s',
                      help="The stable version for the package; if not specified, it is guessed from the current version")
  parser.add_argument("--minor", action='store_true',
                      help="Increase a minor version number from the current version. Please see http://semver.org for reference.")
  parser.add_argument("--major", action='store_true',
                      help="Increase a major version number from the current version. Please see http://semver.org for reference.")
  parser.add_argument("--steps", nargs="+", choices=['tag', 'latest'], default=[
                      'tag', 'latest'], help="Select the steps that you want to execute")
  parser.add_argument("--dry-run", '-q', action='store_true',
                      help="Only print the actions, but do not execute them")
  parser.add_argument("--keep-going", '-f', action='store_true',
                      help="Run all steps, even if some of them fail. HANDLE THIS FLAG WITH CARE!")
  parser.add_argument("--verbose", '-v', action='store_true',
                      help="Print more information")
  parser.add_argument("--force", action='store_true',
                      help="Ignore some checks. Use this with caution.")

  args = parser.parse_args(command_line_options)

  # assert the the version file is there
  version_file = 'version.txt'
  if not os.path.exists(version_file):
    if args.force:
      logger.warn(
          "Could not find the file '%s' containing the version number. Are you inside the root directory of your package?" % version_file)
    else:
      raise ValueError(
          "Could not find the file '%s' containing the version number. Are you inside the root directory of your package?" % version_file)

  # get current version
  current_version = open(version_file).read().rstrip()
  current_version = Version(current_version)
  current_version_list = list(current_version.version)

  if args.stable_version is None:
    stable_version_list = list(current_version_list)
    if args.minor:
      if args.major:
        raise ValueError(
            "--minor and --major should not be specified at the same time.")
      stable_version_list[2] = 0
      stable_version_list[1] += 1
    elif args.major:
      stable_version_list[2] = 0
      stable_version_list[1] = 0
      stable_version_list[0] += 1
    args.stable_version = ".".join("%s" % v for v in stable_version_list)
    print("Assuming stable version to be %s (since current version %s)" %
          (args.stable_version, current_version))
  else:
    if args.minor or args.major:
      raise ValueError(
          "--minor and --major should not be used with --stable-version")

  stable_Version = Version(args.stable_version)
  stable_version_list = list(stable_Version.version)

  if args.latest_version is None:
    # increase current patch version once
    latest_version_list = list(stable_version_list)
    latest_version_list[-1] += 1
    args.latest_version = ".".join([str(v) for v in latest_version_list])
    if current_version.prerelease is not None:
      args.latest_version += "".join(str(p)
                                     for p in current_version.prerelease[:-1]) + '0'
    print("Assuming latest version to be %s (since current version %s)" %
          (args.latest_version, current_version))

  def run_commands(version, *calls):
    """Updates the version.txt to the given version and runs the given
    commands."""
    if version is not None and (args.verbose or args.dry_run):
      print(" - cat '%s' > %s" % (version, version_file))
    if not args.dry_run and version is not None:
      # update version to stable version, if not done yet
      with open(version_file, 'w') as f:
        f.write(version)

    # get all calls
    for call in calls:
      if args.verbose or args.dry_run:
        print(' - ' + ' '.join(call))
      if not args.dry_run:
        # execute call
        if subprocess.call(call):
          # call failed (has non-zero exit status)
          if not args.keep_going:
            raise ValueError("Command '%s' failed; stopping" % ' '.join(call))

  # check the versions
  if args.stable_version is not None and Version(args.latest_version) <= Version(args.stable_version):
    if args.force:
      logger.warn("The latest version '%s' must be greater than the stable version '%s'" % (
          args.latest_version, args.stable_version))
    else:
      raise ValueError("The latest version '%s' must be greater than the stable version '%s'" % (
          args.latest_version, args.stable_version))
  if current_version >= Version(args.latest_version):
    if args.force:
      logger.warn("The latest version '%s' must be greater than the current version '%s'" % (
          args.latest_version, current_version))
    else:
      raise ValueError("The latest version '%s' must be greater than the current version '%s'" % (
          args.latest_version, current_version))
  if args.stable_version is not None and current_version > Version(args.stable_version):
    if args.force:
      logger.warn("The stable version '%s' cannot be smaller than the current version '%s'" % (
          args.stable_version, current_version))
    else:
      raise ValueError("The stable version '%s' cannot be smaller than the current version '%s'" % (
          args.stable_version, current_version))

  if 'tag' in args.steps:
    if args.stable_version is not None and Version(args.stable_version) > current_version:
      print("\nReplacing branch tag in README.rst to '%s'" %
            ('v' + args.stable_version))
      _update_readme(args.stable_version)
      # update stable version on git
      run_commands(args.stable_version, ['git', 'add', 'version.txt', 'README.rst'], [
                   'git', 'commit', '-m', 'Increased stable version to %s' % args.stable_version])
    else:
      # assure that we have the current version
      args.stable_version = current_version
    # add a git tag
    print("\nTagging version '%s'" % args.stable_version)
    run_commands(None, ['git', 'tag', 'v%s' %
                        args.stable_version], ['git', 'push', '--tags'])
    package = os.path.basename(os.path.realpath(os.path.curdir))

  if 'latest' in args.steps:
    # update Gitlab version to latest version
    print("\nReplacing branch tag in README.rst to 'master'")
    _update_readme()
    print("\nSetting latest version '%s'" % args.latest_version)
    run_commands(args.latest_version, ['git', 'add', 'version.txt', 'README.rst'], [
                 'git', 'commit', '-m', 'Increased latest version to %s  [skip ci]' % args.latest_version], ['git', 'push'])

  if 'tag' in args.steps:
    print("\n**IMPORTANT**: Open your web browser and add a changelog here:\n" \
        "  https://gitlab.idiap.ch/bob/%s/tags/v%s/release/edit" % \
        (package, args.stable_version))


if __name__ == '__main__':
  main()
