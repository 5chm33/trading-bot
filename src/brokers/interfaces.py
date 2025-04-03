from abc import ABC, abstractmethod
from typing import Dict, Any

class BrokerInterface(ABC):
    @abstractmethod
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass
        
    @abstractmethod 
    def cancel_all_orders(self) -> None:
        pass
        
    @abstractmethod
    def reconnect(self) -> None:
        pass