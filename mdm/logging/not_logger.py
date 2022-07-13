from typing import Any, Dict

from matplotlib.figure import Figure

from mdm.logging.logger import Logger, Scope


class NotLogger(Logger):

    def setup(self):
        pass

    def teardown(self):
        pass

    def log(self, message: Dict[str, Any], scope: Scope, time_step: int = None):
        pass

    def log_object(self, object: Any, scope: Scope, time_step: int = None):
        pass

    def log_plot(self, figure: Figure, scope: Scope, time_step: int = None):
        pass

