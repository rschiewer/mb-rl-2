from typing import Any, Dict

from matplotlib.figure import Figure

from mdm.logging.logger import Logger, Scope


class NotLogger(Logger):

    def start_session(self):
        pass

    def stop_session(self):
        pass

    def log(self, message: Dict[str, Any], scope: Scope, time_step: int = None):
        pass

    def log_object(self, object: Any, scope: Scope, time_step: int = None):
        pass

    def log_plot(self, figure: Figure, scope: Scope, time_step: int = None):
        pass

