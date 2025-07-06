from enum import auto
from os import path

from mlgame.utils.enum import StringEnum
from mlgame.view.audio_model import SoundProgressSchema, MusicProgressSchema

# game
WIDTH = 950
WIDTH_OF_INFO = 250

HEIGHT = 600
BG_COLOR = "#2B2B49"
PG_COLOR = "#B3E5FC"
SCORE_COLOR_PLUS = "#76ff03"
SCORE_COLOR_MINUS = "#ec407a"

# ball -> squid
# BALL_COLOR = "#FFEB3B"
SQUID_W = 40
SQUID_H = 60
LEVEL_THRESHOLDS = [10, 30, 60, 100, 150, 200]
LEVEL_PROPERTIES = {
    1: {'size_ratio': 1.0, 'vel': 10},
    2: {'size_ratio': 1.2, 'vel': 12},
    3: {'size_ratio': 1.4, 'vel': 15},
    4: {'size_ratio': 1.6, 'vel': 18},
    5: {'size_ratio': 1.8, 'vel': 21},
    6: {'size_ratio': 2.0, 'vel': 25},
}

ASSET_IMAGE_DIR = path.join(path.dirname(__file__), "../asset/img")


# food
class FoodTypeEnum(StringEnum):
    FOOD_1 = auto()
    FOOD_2 = auto()
    FOOD_3 = auto()
    GARBAGE_1 = auto()
    GARBAGE_2 = auto()
    GARBAGE_3 = auto()


FOOD_LV1_SIZE = 30
FOOD_LV2_SIZE = 40
FOOD_LV3_SIZE = 50

# path of assets
ASSET_PATH = path.join(path.dirname(__file__), "..", "asset")
LEVEL_PATH = path.join(path.dirname(__file__), "..", "levels")
SOUND_PATH = path.join(path.dirname(__file__), "..", "asset", "sounds")
MUSIC_PATH = path.join(path.dirname(__file__), "..", "asset", "music")

BGM01_PATH = path.join(MUSIC_PATH, "bgm01.mp3")
BG_PATH = path.join(ASSET_IMAGE_DIR, "background.png")
SQUID_PATH = path.join(ASSET_IMAGE_DIR, "squid.png")
EATING_GOOD_PATH = path.join(SOUND_PATH, "eat_good_food.mp3")
EATING_BAD_PATH = path.join(SOUND_PATH, "eat_bad_food.mp3")
PASS_PATH = path.join(SOUND_PATH, "pass.mp3")
FAIL_PATH = path.join(SOUND_PATH, "fail.mp3")
LV_UP_PATH = path.join(SOUND_PATH, "lv_up.mp3")
LV_DOWN_PATH = path.join(SOUND_PATH, "lv_down.mp3")
COLLISION_PATH = path.join(SOUND_PATH, "collision.mp3")

IMG_ID_FOOD01_L = "food_01_L"
IMG_ID_FOOD02_L = "food_02_L"
IMG_ID_FOOD03_L = "food_03_L"
IMG_ID_FOOD01_R = "food_01_R"
IMG_ID_FOOD02_R = "food_02_R"
IMG_ID_FOOD03_R = "food_03_R"

FOOD01_L_PATH = path.join(ASSET_IMAGE_DIR, "food_01_L.png")
FOOD02_L_PATH = path.join(ASSET_IMAGE_DIR, "food_02_L.png")
FOOD03_L_PATH = path.join(ASSET_IMAGE_DIR, "food_03_L.png")
FOOD01_R_PATH = path.join(ASSET_IMAGE_DIR, "food_01_R.png")
FOOD02_R_PATH = path.join(ASSET_IMAGE_DIR, "food_02_R.png")
FOOD03_R_PATH = path.join(ASSET_IMAGE_DIR, "food_03_R.png")

GARBAGE01_PATH = path.join(ASSET_IMAGE_DIR, "garbage_01.png")
GARBAGE02_PATH = path.join(ASSET_IMAGE_DIR, "garbage_02.png")
GARBAGE03_PATH = path.join(ASSET_IMAGE_DIR, "garbage_03.png")

ASSET_IMG_URL = "https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid/main/asset/img/"
BG_URL = ASSET_IMG_URL + "background.png"
SQUID_URL = ASSET_IMG_URL + "squid.png"
# Food URLs
FOOD01_L_URL = ASSET_IMG_URL + "food_01_L.png"
FOOD02_L_URL = ASSET_IMG_URL + "food_02_L.png"  # Assuming the naming pattern is similar
FOOD03_L_URL = ASSET_IMG_URL + "food_03_L.png"
FOOD01_R_URL = ASSET_IMG_URL + "food_01_R.png"
FOOD02_R_URL = ASSET_IMG_URL + "food_02_R.png"
FOOD03_R_URL = ASSET_IMG_URL + "food_03_R.png"

# Garbage URLs
GARBAGE01_URL = ASSET_IMG_URL + "garbage_01.png"
GARBAGE02_URL = ASSET_IMG_URL + "garbage_02.png"
GARBAGE03_URL = ASSET_IMG_URL + "garbage_03.png"
# BAR_URL = "https://raw.githubusercontent.com/PAIA/dont_touch/master/asset/image/bar.png"

# https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/easy_game/main/asset/img/background.jpg

# Music URL
MUSIC_URL = "https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid/main/asset/music/"
SOUND_URL = "https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid/main/asset/sounds/"

BGM01_URL = MUSIC_URL + "bgm01.mp3"

# Sound URLs
EATING_GOOD_URL = SOUND_URL + "eat_good_food.mp3"
EATING_BAD_URL = SOUND_URL + "eat_bad_food.mp3"
PASS_URL = SOUND_URL + "pass.mp3"
FAIL_URL = SOUND_URL + "fail.mp3"
LV_UP_URL = SOUND_URL + "lv_up.mp3"
LV_DOWN_URL = SOUND_URL + "lv_down.mp3"
COLLISION_URL = SOUND_URL + "collision.mp3"

BGM01_OBJ = MusicProgressSchema(music_id=f"bgm01").__dict__
EATING_GOOD_OBJ = SoundProgressSchema(sound_id='eat_good_food').__dict__
EATING_BAD_OBJ = SoundProgressSchema(sound_id='eat_bad_food').__dict__
PASS_OBJ = SoundProgressSchema(sound_id='pass').__dict__
FAIL_OBJ = SoundProgressSchema(sound_id='fail').__dict__
LV_UP_OBJ = SoundProgressSchema(sound_id='lv_up').__dict__
LV_DOWN_OBJ = SoundProgressSchema(sound_id='lv_down').__dict__
COLLISION_OBJ = SoundProgressSchema(sound_id='collision').__dict__
