from typing import Dict, Any
import os

import neptune.new as neptune
from neptune.new.run import Run

from mdm.logging.logger import Logger, Scope


class NeptuneLogger(Logger):

    def __init__(self, project: str,
                 run: Run = None,
                 api_token: str = None):
        self.project = project
        self._token = api_token
        self._run = run

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
        if self._run:
            self.teardown()
        self._run = neptune.init(project=self.project, api_token=self.token)

    def teardown(self):
        if self._run:
            self._run.stop()

    def log(self, message: Dict[str, Any], scope: Scope, time_step: int = None):
        for name, value in message:
            self._run[f'{scope}/{name}'].log(value)

