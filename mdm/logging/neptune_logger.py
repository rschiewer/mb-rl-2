import io
import shutil
from typing import Dict, Any, Union
import os
from pathlib import Path
import time
import asyncio
from threading import Thread

import neptune
from matplotlib.figure import Figure
from neptune import Run
from neptune.types import File
import numpy as np
from PIL import Image

from mdm.logging.logger import Logger, Scope
from mdm.utils.utils import InMemoryFile


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
                self._run = neptune.init_run(project=self._project, run=self._run_id, api_token=self.token,
                                         capture_stdout=False, capture_stderr=False)
            else:
                self._run = neptune.init_run(project=self._project, api_token=self.token, capture_stdout=False,
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
            elif isinstance(value, (Path, InMemoryFile)):
                self.log_file(value, full_scope, time_step=time_step)
            #elif isinstance(value, str) and Path(value).exists():
            #    self.log_file(value, full_scope, time_step=time_step)
            else:
                self._run[str(full_scope)].log(value, step=time_step)

    def log_object(self, object: Any, scope: Scope, time_step: int = None):
        if isinstance(object, io.BytesIO):
            timestamp = time.time_ns()
            pid = os.getpid()
            tmp_file_name = f'.{pid}_{timestamp}'
            with open(tmp_file_name, 'wb') as f:
                f.write(object.read())
            self._run[str(scope)].log(tmp_file_name, step=time_step)
            os.remove(tmp_file_name)
        else:
            raise ValueError('Unknown type for logging')

    def log_file(self,
                 path: str | Path | InMemoryFile,
                 scope: Scope,
                 time_step: int = None):
        if isinstance(path, (str, Path)):
            path = InMemoryFile(path)
            #path = str(path)
            #i_start = path.rindex('/') + 1 if '/' in path else 0
            #fname = path[i_start:]
            #if time_step:
            #    fname = f'{time_step}_{fname}'
            #self._run[str(scope / fname)].upload(path, wait=True)
        #elif isinstance(path, InMemoryFile):
        #    stream_file = File.from_stream(path.buffer, extension=path.extension)
        #    self._run[str(scope)].upload(stream_file, wait=False)
        #else:
        #    raise ValueError(f'Unsupported resource format for upload: {type(path)}')

        if time_step is not None:
            path.name += f'_{time_step}'
        scope /= path.name

        stream_file = File.from_stream(path.buffer, extension=path.extension)
        self._run[str(scope)].upload(stream_file, wait=False)

        #self._run[str(scope)].append(str(new_path), wait=True)

    def log_plot(self, figure: Image, scope: Union[Scope, str], time_step: int = None):
        self._run[str(scope)].log(figure, step=time_step)


