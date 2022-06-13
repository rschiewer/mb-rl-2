from typing import Dict, Any, Union
import os

import neptune.new as neptune
from matplotlib.figure import Figure
from neptune.new.run import Run
import numpy as np

from mdm.logging.logger import Logger, Scope


class NeptuneLogger(Logger):

    def __init__(self,
                 project: str,
                 run: Run = None,
                 api_token: str = None):
        self._project = project
        self._run = run
        self._token = api_token

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self,
                new_project):
        if self._run:
            raise ValueError('Can\'t change project during an active run')
        self._project = new_project

    @property
    def token(self):
        if self._token:
            return self._token
        elif os.environ['NEPTUNE_API_TOKEN']:
            return os.environ['NEPTUNE_API_TOKEN']
        else:
            raise RuntimeError('Please specify NEPTUNE_API_TOKEN environment variable or provide the api token via '
                               'constructor argument')

    def setup(self):
        if not self._run:
            self._run = neptune.init(project=self._project, api_token=self.token)

    def teardown(self):
        if self._run:
            self._run.stop()

    def log(self,
            message: Dict[str, Any],
            scope: Scope,
            time_step: int = None):
        for name, value in message.items():
            full_scope = scope / name
            if isinstance(value, np.ndarray):
                value = value.flatten()
                for v in value:
                    self._run[str(full_scope)].log(v, step=time_step)
            elif isinstance(value, dict):
                self.log(value, full_scope, time_step=time_step)
            else:
                self._run[str(full_scope)].log(value, step=time_step)

    def log_object(self, object: Any, scope: Union[Scope, str], time_step: int = None):
        pass

    def log_plot(self, figure: Figure, scope: Union[Scope, str], time_step: int = None):
        pass
