import sys
from os import path
sys.path.append(path.dirname(__file__))

from src.game import SwimmingSquid

GAME_SETUP = {
    "game": SwimmingSquid,
    # "dynamic_ml_clients":True
}
