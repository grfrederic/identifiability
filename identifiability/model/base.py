from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def run(self):
        """Run the model."""
        pass

    @abstractmethod
    def prior_log_prob(self, parameters):
        """Log-probability of parameters."""
        pass

