'''Tests for the global bob's configuration functionality'''

from .rc_config import _loadrc, ENVNAME
from .scripts import main_cli
from click.testing import CliRunner
import os
import pkg_resources
path = pkg_resources.resource_filename('bob.extension', 'data')


def test_rc_env():

  os.environ[ENVNAME] = os.path.join(path, 'defaults-config')
  c = _loadrc()  # should load from environment variable
  REFERENCE = {
      "bob.db.atnt.directory": "/home/bob/databases/atnt",
      "bob.db.mobio.directory": "/home/bob/databases/mobio"
  }

  assert c == REFERENCE
  assert c['random'] is None


def test_bob_config():
  defaults_config = os.path.join(path, 'defaults-config')
  runner = CliRunner(env={ENVNAME: defaults_config})

  # test config show
  result = runner.invoke(main_cli, ['config', 'show'])
  assert result.exit_code == 0, result.exit_code
  assert 'defaults-config' in result.output, result.output
  assert open(defaults_config).read() in result.output, result.output

  # test config get (existing key)
  result = runner.invoke(main_cli,
                         ['config', 'get', 'bob.db.atnt.directory'])
  assert result.exit_code == 0, result.exit_code
  assert result.output == '/home/bob/databases/atnt\n', result.output

  # test config get (non-existing key)
  result = runner.invoke(main_cli, ['config', 'get', 'bob.db.atnt'])
  assert result.exit_code == 1, result.exit_code

  # test config set
  runner = CliRunner()
  with runner.isolated_filesystem():
    bobrcfile = 'bobrc'
    result = runner.invoke(
        main_cli, [
            'config', 'set', 'bob.db.atnt.directory',
            '/home/bob/databases/orl_faces'
        ],
        env={
            ENVNAME: bobrcfile
        })
    assert result.exit_code == 0, result.exit_code

    # read the config back to make sure it is ok.
    result = runner.invoke(
        main_cli, ['config', 'show'], env={
            ENVNAME: bobrcfile
        })
    assert result.exit_code == 0, result.exit_code
    expected_output = '''Displaying `bobrc':
{
    "bob.db.atnt.directory": "/home/bob/databases/orl_faces"
}
'''
    assert expected_output == result.output, result.output
