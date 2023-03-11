from typing import Dict, Any, Union
import os
from pathlib import Path

from neptune.new.types import File
import neptune.new as neptune
from matplotlib.figure import Figure
from neptune.new.run import Run, InactiveRunException
import numpy as np
from PIL import Image

from mdm.logging.logger import Logger, Scope


class NeptuneLogger(Logger):

    def __init__(self,
                 project: str,
                 run_handler: Run = None,
                 run_id: str = None,
                 api_token: str = None):
        super(NeptuneLogger, self).__init__()
        self._project = project
        self._run = run_handler
        self._token = api_token
        self._run_id = run_id

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

    def start_session(self):
        if not self._run:
            if self._run_id:
                self._run = neptune.init(project=self._project, run=self._run_id, api_token=self.token,
                                         capture_stdout=False, capture_stderr=False)
            else:
                self._run = neptune.init(project=self._project, api_token=self.token, capture_stdout=False,
                                         capture_stderr=False)
                self._run_id = self._run['sys/id'].fetch()
                if self._run_id.startswith('https'):
                    i_start = self._run_id.rindex('/')
                    self._run_id = self._run_id[i_start+1:]

    def stop_session(self):
        if self._run:
            self._run.wait()
            self._run.stop()
            self._run = None

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
            elif isinstance(value, list):
                for v in value:
                    self.log({name: v}, scope, time_step=time_step)
            elif isinstance(value, Figure):
                self.log_plot(value, full_scope, time_step=time_step)
            else:
                self._run[str(full_scope)].log(value, step=time_step)

    def log_object(self, object: Any, scope: Union[Scope, str], time_step: int = None):
        raise NotImplementedError('Directly logging objects is not supported')

    def log_file(self,
                 path: Union[str, Path],
                 scope: Scope,
                 time_step: int = None):
        self._run[str(scope)].upload(str(path))

    def log_plot(self, figure: Image, scope: Union[Scope, str], time_step: int = None):
        self._run[str(scope)].log(figure)


