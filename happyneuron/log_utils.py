"""Logging utilities for a common interface.

Classes
-------
HNLogger
    Simple logging to stdout with formatting.
"""

import logging
import sys


class HNLogger(object):
    """Logging for HappyNeuron operations.

    Parameters
    ----------
    name : str
        Name of the application. Default: 'happyneuron'
    format_string : str
        Formatting string for the logs. By default, includes the timestamp,
        application name, and message. May be modified to include extra values,
        for example MPI rank.
    extra_vals : dict
        Additional values to include automatically in log messages.
    """

    DEFAULT_FORMAT = '%(asctime)s %(name)s : %(message)s'

    def __init__(self, name='happyneuron', format_string=None,
                 extra_vals=None, quiet=False):
        self.logger = logging.getLogger('img_to_cloudvolume.py')
        self.__logger = self.logger
        self.logger.setLevel(logging.INFO)
        self.format_string = format_string if format_string is not None \
                             else self.DEFAULT_FORMAT
        self.handler = None



        if extra_vals is not None:
            self.logger = logging.LoggerAdapter(self.logger, extra_vals)

    def info(self, msg):
        """Write an info-level log."""
        self.logger.info(msg)

    def quiet(self):
        """Turn off logging."""
        if self.handler is not None:
            self.__logger.removeHandler(self.handler)

        self.handler = logging.NullHandler()
        self.__logger.addHandler(syslog)

    def verbose(self):
        """Turn on logging to stdout."""
        if self.handler is not None:
            self.__logger.removeHandler(self.handler)

        self.handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(self.format_string)
        self.handler.setFormatter(formatter)
        self.__logger.addHandler(self.handler)
