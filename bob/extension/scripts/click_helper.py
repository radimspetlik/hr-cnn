from ..log import set_verbosity_level
from ..config import load, mod_to_context
import click
import logging
# This needs to bob so that logger is configured for all bob packages.
logger = logging.getLogger('bob')
try:
  basestring
except NameError:
  basestring = str


def verbosity_option(**kwargs):
  def custom_verbosity_option(f):
    def callback(ctx, param, value):
      ctx.meta['verbosity'] = value
      set_verbosity_level(logger, value)
      logger.debug("Logging of the `bob' logger was set to %d", value)
      return value
    return click.option(
        '-v', '--verbose', count=True,
        expose_value=False,
        help="Increase the verbosity level from 0 (only error messages) to 1 "
        "(warnings), 2 (log messages), 3 (debug information) by adding the "
        "--verbose option as often as desired (e.g. '-vvv' for debug).",
        callback=callback, **kwargs)(f)
  return custom_verbosity_option


class ConfigCommand(click.Command):
  """A click.Command that can take options both form command line options and
  configuration files. In order to use this class, you have to use the
  :any:`Option` class also.

  Attributes
  ----------
  config_argument_name : TYPE
      Description
  entry_point_group : TYPE
      Description
  """

  def __init__(self, name, context_settings=None, callback=None, params=None,
               help=None, epilog=None, short_help=None,
               options_metavar='[OPTIONS]',
               add_help_option=True, entry_point_group=None,
               config_argument_name='CONFIG', **kwargs):
    self.config_argument_name = config_argument_name
    self.entry_point_group = entry_point_group
    click.Command.__init__(
        self, name, context_settings=context_settings, callback=callback,
        params=params, help=help, epilog=epilog, short_help=short_help,
        options_metavar=options_metavar, add_help_option=add_help_option,
        **kwargs)
    # Add the config argument to the command
    click.argument(config_argument_name, nargs=-1)(self)

  def invoke(self, ctx):
    config_files = ctx.params[self.config_argument_name.lower()]
    # load and normalize context from config files
    config_context = load(
        config_files, entry_point_group=self.entry_point_group)
    config_context = mod_to_context(config_context)
    for param in self.params:
      if param.name not in ctx.params:
        continue
      value = ctx.params[param.name]
      if not hasattr(param, 'user_provided'):
        if value == param.default:
          param.user_provided = False
        else:
          param.user_provided = True
      if not param.user_provided and param.name in config_context:
        ctx.params[param.name] = param.full_process_value(
            ctx, config_context[param.name])
      # raise exceptions if the value is required.
      if hasattr(param, 'real_required'):
        param.required = param.real_required
        ctx.params[param.name] = param.full_process_value(
            ctx, ctx.params[param.name])

    return super(ConfigCommand, self).invoke(ctx)


class ResourceOption(click.Option):
  """A click.Option that is aware if the user actually provided this option
  through command-line or it holds a default value. The option can also be a
  resource that will be automatically loaded.

  Attributes
  ----------
  entry_point_group : str or None
      If provided, the strings values to this option are assumed to be entry
      points from ``entry_point_group`` that need to be loaded.
  real_required : bool
      Holds the real value of ``required`` here. The ``required`` value is
      hidden from click since the option may be loaded later through the
      configuration files.
  user_provided : bool
      True if the user actually provided this option through command-line or
      using environment variables.
  """

  def __init__(self, param_decls=None, show_default=False, prompt=False,
               confirmation_prompt=False, hide_input=False, is_flag=None,
               flag_value=None, multiple=False, count=False,
               allow_from_autoenv=True, type=None, help=None,
               entry_point_group=None, required=False, **kwargs):
    self.entry_point_group = entry_point_group
    self.real_required = required
    kwargs['required'] = False
    click.Option.__init__(
        self, param_decls=param_decls, show_default=show_default,
        prompt=prompt, confirmation_prompt=confirmation_prompt,
        hide_input=hide_input, is_flag=is_flag, flag_value=flag_value,
        multiple=multiple, count=count, allow_from_autoenv=allow_from_autoenv,
        type=type, help=help, **kwargs)

  def consume_value(self, ctx, opts):
    value = opts.get(self.name)
    self.user_provided = True
    if value is None:
      value = ctx.lookup_default(self.name)
      self.user_provided = False
    if value is None:
      value = self.value_from_envvar(ctx)
      if value is not None:
        self.user_provided = True
    return value

  def full_process_value(self, ctx, value):
    value = super(ResourceOption,
                  self).full_process_value(ctx, value)

    if self.entry_point_group is not None:
      keyword = self.entry_point_group.split('.')[-1]
      while isinstance(value, basestring):
        value = load([value], entry_point_group=self.entry_point_group)
        value = getattr(value, keyword)

    return value
