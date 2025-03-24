from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from datetime import time
import json

class ParkingCase(Enum):
    NO_INFO = 0        # When we don't have information
    NO_PARKING = 1     # When parking is not allowed
    PARKING_ALLOWED = 2 # When parking is allowed

@dataclass
class TimeRestriction:
    start_time: time
    end_time: time
    enforcement_days: List[str]
    special_notes: Optional[str] = None

    def to_dict(self):
        return {
            "start_time": self.start_time.strftime("%H:%M"),
            "end_time": self.end_time.strftime("%H:%M"),
            "enforcement_days": self.enforcement_days,
            "special_notes": self.special_notes
        }

@dataclass
class ParkingLot:
    name: str
    location: Optional[str] = None
    capacity: Optional[int] = None

    def to_dict(self):
        return {
            "name": self.name,
            "location": self.location,
            "capacity": self.capacity
        }

@dataclass
class AdditionalInfo:
    lots: List[ParkingLot]
    time_restriction: Optional[TimeRestriction] = None
    alternative_lots: Optional[List[ParkingLot]] = None
    reason: Optional[str] = None

    def to_dict(self):
        return {
            "lots": [lot.to_dict() for lot in self.lots],
            "time_restriction": self.time_restriction.to_dict() if self.time_restriction else None,
            "alternative_lots": [lot.to_dict() for lot in (self.alternative_lots or [])],
            "reason": self.reason
        }

class ParkingResponse:
    def __init__(self, 
                 case: ParkingCase, 
                 main_message: str, 
                 additional_info: AdditionalInfo):
        self.case = case
        self.main_message = main_message
        self.additional_info = additional_info

    def to_dict(self):
        return {
            "case": self.case.value,
            "main_message": self.main_message,
            "additional_info": self.additional_info.to_dict()
        }