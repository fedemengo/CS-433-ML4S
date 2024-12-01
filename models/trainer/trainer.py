from abc import ABC, abstractmethod
import optuna

class BaseTrainer(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def evaluate(self, X_val, y_val):
        pass
    
    @abstractmethod
    def get_optuna_params(self, trial):
        pass
        