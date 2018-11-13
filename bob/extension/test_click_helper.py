import click
from click.testing import CliRunner
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)


def test_verbosity_option():

    for VERBOSITY, OPTIONS in zip([0, 1, 2, 3],
                                  [[], ['-v'], ['-vv'], ['-vvv']]):
        @click.command()
        @verbosity_option()
        def cli():
            ctx = click.get_current_context()
            verbose = ctx.meta['verbosity']
            assert verbose == VERBOSITY, verbose

        runner = CliRunner()
        result = runner.invoke(cli, OPTIONS, catch_exceptions=False)
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_commands_with_config_1():
    # random test
    @click.command(
        cls=ConfigCommand, entry_point_group='bob.extension.test_config_load')
    def cli(**kwargs):
        pass

    runner = CliRunner()
    result = runner.invoke(cli, ['basic_config'])
    assert result.exit_code == 0, (result.exit_code, result.output)


def test_commands_with_config_2():
    # test option with valid default value
    @click.command(
        cls=ConfigCommand, entry_point_group='bob.extension.test_config_load')
    @click.option(
        '-a', cls=ResourceOption, default=3)
    def cli(a, **kwargs):
        click.echo('{}'.format(a))

    runner = CliRunner()

    result = runner.invoke(cli, [])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '3', result.output

    result = runner.invoke(cli, ['basic_config'])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '1', result.output

    result = runner.invoke(cli, ['-a', 2])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '2', result.output

    result = runner.invoke(cli, ['-a', 3, 'basic_config'])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '3', result.output

    result = runner.invoke(cli, ['basic_config', '-a', 3])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '3', result.output


def test_commands_with_config_3():
    # test required options
    @click.command(
        cls=ConfigCommand, entry_point_group='bob.extension.test_config_load')
    @click.option(
        '-a', cls=ResourceOption, required=True)
    def cli(a, **kwargs):
        click.echo('{}'.format(a))

    runner = CliRunner()

    result = runner.invoke(cli, ['basic_config'])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '1', result.output

    result = runner.invoke(cli, ['-a', 2])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '2', result.output

    result = runner.invoke(cli, ['-a', 3, 'basic_config'])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '3', result.output

    result = runner.invoke(cli, ['basic_config', '-a', 3])
    assert result.exit_code == 0, (result.exit_code, result.output)
    assert result.output.strip() == '3', result.output

    # somehow this test breaks the following tests so I test it last
    result = runner.invoke(cli, [])
    assert result.exit_code == 2, (result.exit_code, result.output)
