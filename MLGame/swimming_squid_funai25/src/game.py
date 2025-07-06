import copy
import json
import time
import os.path

import pygame

from mlgame.game.paia_game import PaiaGame, GameResultState, GameStatus
from mlgame.utils.enum import get_ai_name
from mlgame.view.audio_model import create_music_init_data, create_sound_init_data
from mlgame.view.decorator import check_game_progress, check_game_result
from mlgame.view.view_model import *
from .foods import *
from .game_object import Squid, LevelParams, ScoreText, WindowConfig


def revise_squid_coordinate(squid: Squid, playground: pygame.Rect):
    ball_rect = copy.deepcopy(squid.rect)
    if ball_rect.left < playground.left:
        ball_rect.left = playground.left
    elif ball_rect.right > playground.right:
        ball_rect.right = playground.right

    if ball_rect.top < playground.top:
        ball_rect.top = playground.top
    elif ball_rect.bottom > playground.bottom:
        ball_rect.bottom = playground.bottom
    squid.rect = ball_rect
    pass


class SwimmingSquid(PaiaGame):
    """
    This is a Interface of a game
    """

    def __init__(
            self,
            level: int = -1,
            level_file: str = "",
            contest: int = 0,
            *args, **kwargs):
        super().__init__(user_num=1)
        self.game_result_state = GameResultState.UN_PASSED
        self.scene = Scene(width=WIDTH, height=HEIGHT, color=BG_COLOR, bias_x=0, bias_y=0)
        self._level = level
        self._level_file = level_file
        self.foods = pygame.sprite.Group()
        self._help_texts = pygame.sprite.Group()

        self._game_params = None
        self._sounds = []
        self.contest = contest
        self.timerDuration = 300
        self._init_game()
        self.start_time  = time.time()

    def _init_game_by_file(self, level_file_path: str):
        try:
            with open(level_file_path) as f:
                game_params = LevelParams(**json.load(f))
                self._used_file = level_file_path

        except:
            # If the file doesn't exist, use default parameters
            print("此關卡檔案不存在，遊戲將會會自動使用第一關檔案 001.json。")
            print("This level file is not existed , game will load 001.json automatically.")
            with open(os.path.join(LEVEL_PATH, "001.json")) as f:
                game_params = LevelParams(**json.load(f))
                self._level = 1
                self._level_file = ""
        finally:
            # set game params
            self.playground = pygame.Rect(
                0, 0,
                game_params.playground_size_w,
                game_params.playground_size_h
            )

            self._score_to_pass = game_params.score_to_pass
            self._frame_limit = game_params.time_to_play
            self.playground.center = ((WIDTH - WIDTH_OF_INFO) / 2, HEIGHT / 2)
            self._food_window = WindowConfig(
                left=self.playground.left, right=self.playground.right,
                top=self.playground.top, bottom=self.playground.bottom)
            self._garbage_window = WindowConfig(
                left=self.playground.left, right=self.playground.right,
                top=self.playground.top - 60, bottom=self.playground.top - 10)
            self._food_pos_list = []
            self._garbage_pos_list = []
            # init game
            self.squid = Squid()
            self.foods.empty()
            self._create_foods(Food1, game_params.food_1)
            self._create_foods(Food2, game_params.food_2)
            self._create_foods(Food3, game_params.food_3)
            self._create_foods(Garbage1, game_params.garbage_1)
            self._create_foods(Garbage2, game_params.garbage_2)
            self._create_foods(Garbage3, game_params.garbage_3)

            self.frame_count = 0
            self._frame_count_down = self._frame_limit
            game_params.top = self.playground.top
            game_params.left = self.playground.left
            game_params.bottom = self.playground.bottom
            game_params.right = self.playground.right
            self._game_params = game_params

    def update(self, commands):
        # handle command
        ai_1p_cmd = commands[get_ai_name(0)]
        if ai_1p_cmd is not None:
            action = ai_1p_cmd[0]
        else:
            action = "NONE"
        if(time.time()- self.start_time > self.timerDuration and self.contest):   
            self.squid.update(["NONE"])
        else:
        
            self.squid.update(action)
            revise_squid_coordinate(self.squid, self.playground)
            # update sprite
            self.foods.update(playground=self.playground, squid=self.squid)
            self._help_texts.update()
            # handle collision

            self._check_foods_collision()
            # self._timer = round(time.time() - self._begin_time, 3)

            self.frame_count += 1
            self._frame_count_down = self._frame_limit - self.frame_count
            # self.draw()

        if not self.is_running:
            return "RESET"

    def _check_foods_collision(self):
        hits = pygame.sprite.spritecollide(self.squid, self.foods, True)
        if hits:
            for food in hits:
                # self.ball.score += food.score
                # growth play special sound
                self.squid.eat_food_and_change_level_and_play_sound(food, self._sounds)
                self._create_foods(food.__class__, 1)
                if isinstance(food, (Food1, Food2, Food3,)):
                    # add help text
                    ScoreText(
                        text=f"+{food.score}",
                        color=SCORE_COLOR_PLUS,
                        x=food.rect.centerx,
                        y=food.rect.centery,
                        groups=self._help_texts
                    )

                    self._sounds.append(EATING_GOOD_OBJ)
                elif isinstance(food, (Garbage1, Garbage2, Garbage3,)):
                    # add help text
                    ScoreText(
                        text=f"{food.score}",
                        color=SCORE_COLOR_MINUS,
                        x=food.rect.centerx,
                        y=food.rect.centery,
                        groups=self._help_texts
                    )
                    self._sounds.append(EATING_BAD_OBJ)

    def get_data_from_game_to_player(self):
        """
        send something to game AI
        we could send different data to different ai
        """
        to_players_data = {}
        foods_data = [{"x": food.rect.centerx, "y": food.rect.centery,
                       "w": food.rect.width, "h": food.rect.height,
                       "type": str(food.type), "score": food.score} for food in self.foods]

        data_to_1p = {
            "frame": self.frame_count,
            "self_x": self.squid.rect.centerx,
            "self_y": self.squid.rect.centery,
            "self_w": self.squid.rect.width,
            "self_h": self.squid.rect.height,
            "self_vel": self.squid.vel,
            "self_lv": self.squid.lv,
            "foods": foods_data,
            "score": self.squid.score,
            "score_to_pass": self._score_to_pass,
            "status": self.get_game_status(),
            "env": self._game_params.__dict__

        }

        to_players_data[get_ai_name(0)] = data_to_1p
        # should be equal to config. GAME_SETUP["ml_clients"][0]["name"]

        return to_players_data

    def get_game_status(self):

        if self.is_running:
            status = GameStatus.GAME_ALIVE
        elif self.is_passed:
            status = GameStatus.GAME_PASS
        else:
            status = GameStatus.GAME_OVER
        return status

    def reset(self):

        if self.is_passed:
            self._level += 1
            self._sounds.append(PASS_OBJ)
        else:
            self._sounds.append(FAIL_OBJ)

        self._init_game()

        pass

    def _init_game(self):
        if path.isfile(self._level_file):
            # set by injected file
            self._init_game_by_file(self._level_file)
            pass
        else:
            level_file_path = os.path.join(LEVEL_PATH, f"{self._level:03d}.json")
            self._init_game_by_file(level_file_path)

    @property
    def is_passed(self):
        return self.squid.score >= self._score_to_pass

    @property
    def is_running(self):
        # return self.frame_count < self._frame_limit
        return (self.frame_count < self._frame_limit) and (not self.is_passed)

    def get_scene_init_data(self):
        """
        Get the initial scene and object information for drawing on the web
        """
        # bg_path = path.join(ASSET_PATH, "img/background.jpg")
        # background = create_asset_init_data(
        #     "background", WIDTH, HEIGHT, bg_path,
        #     github_raw_url="https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/easy_game/main/asset/img/background.jpg")

        scene_init_data = {
            "scene": self.scene.__dict__,
            "assets": [
                create_asset_init_data("bg", 1000, 1000, BG_PATH, BG_URL),
                create_asset_init_data("squid", SQUID_W, SQUID_H, SQUID_PATH, SQUID_URL),
                create_asset_init_data(IMG_ID_FOOD01_L, FOOD_LV1_SIZE, FOOD_LV1_SIZE, FOOD01_L_PATH, FOOD01_L_URL),
                create_asset_init_data(IMG_ID_FOOD02_L, FOOD_LV2_SIZE, FOOD_LV2_SIZE, FOOD02_L_PATH, FOOD02_L_URL),
                create_asset_init_data(IMG_ID_FOOD03_L, FOOD_LV3_SIZE, FOOD_LV3_SIZE, FOOD03_L_PATH, FOOD03_L_URL),
                create_asset_init_data(IMG_ID_FOOD01_R, FOOD_LV1_SIZE, FOOD_LV1_SIZE, FOOD01_R_PATH, FOOD01_R_URL),
                create_asset_init_data(IMG_ID_FOOD02_R, FOOD_LV2_SIZE, FOOD_LV2_SIZE, FOOD02_R_PATH, FOOD02_R_URL),
                create_asset_init_data(IMG_ID_FOOD03_R, FOOD_LV3_SIZE, FOOD_LV3_SIZE, FOOD03_R_PATH, FOOD03_R_URL),
                create_asset_init_data("garbage01", FOOD_LV1_SIZE, FOOD_LV1_SIZE, GARBAGE01_PATH, GARBAGE01_URL),
                create_asset_init_data("garbage02", FOOD_LV2_SIZE, FOOD_LV2_SIZE, GARBAGE02_PATH, GARBAGE02_URL),
                create_asset_init_data("garbage03", FOOD_LV3_SIZE, FOOD_LV3_SIZE, GARBAGE03_PATH, GARBAGE03_URL),
            ],
            "background": [
                # create_image_view_data(
                #     'bg', self.playground.x, self.playground.y,
                #     self.playground.w, self.playground.h)
            ],
            "musics": [
                create_music_init_data("bgm01", file_path=BGM01_PATH, github_raw_url=BGM01_URL),

            ],
            # Create the sounds list using create_sound_init_data
            "sounds": [
                create_sound_init_data("eat_good_food", file_path=EATING_GOOD_PATH, github_raw_url=EATING_GOOD_URL),
                create_sound_init_data("eat_bad_food", file_path=EATING_BAD_PATH, github_raw_url=EATING_BAD_URL),
                create_sound_init_data("pass", file_path=PASS_PATH, github_raw_url=PASS_URL),
                create_sound_init_data("fail", file_path=FAIL_PATH, github_raw_url=FAIL_URL),
                create_sound_init_data("lv_up", file_path=LV_UP_PATH, github_raw_url=LV_UP_URL),
                create_sound_init_data("lv_down", file_path=LV_DOWN_PATH, github_raw_url=LV_DOWN_URL),
            ]}
        return scene_init_data

    @check_game_progress
    def get_scene_progress_data(self):
        """
        Get the position of game objects for drawing on the web
        """
        foods_data = []
        for food in self.foods:
            foods_data.append(food.game_object_data)
        game_obj_list = [self.squid.game_object_data]
        help_texts = [
            obj.game_object_data for obj in self._help_texts
        ]
        game_obj_list.extend(foods_data)
        game_obj_list.extend(help_texts)

        backgrounds = [
            # create_image_view_data("background", 0, 0, WIDTH, HEIGHT),
            # create_rect_view_data(
            #     "playground", self.playground.x, self.playground.y,
            #     self.playground.w, self.playground.h, PG_COLOR)
            create_image_view_data(
                'bg', self.playground.x, self.playground.y,
                self.playground.w, self.playground.h)
        ]
        foregrounds = [

        ]
        star_string = '+' * self.squid.lv
        toggle_objs = [
            create_text_view_data(f"Squid Lv: {star_string}", 705, 50, "#EEEEEE", "20px Consolas BOLD"),
            create_text_view_data(f"To Lv up: {LEVEL_THRESHOLDS[self.squid.lv - 1] - self.squid.score :04d} pt", 705,
                                  80, "#EEEEEE", "20px Consolas BOLD"),
            create_text_view_data(f"File :{os.path.basename(self._used_file)}", 705, 110, "#EEEEEE",
                                  "20px Consolas BOLD"),
            create_text_view_data(f"Vel     : {self.squid.vel:4d}", 705, 170, "#EEEEEE", "20px Consolas BOLD"),
            create_text_view_data(f"Timer   : {self._frame_count_down:04d}", 705, 200, "#EEEEEE", "20px Consolas BOLD"),
            create_text_view_data(f"My Score: {self.squid.score:04d} pt", 705, 230, "#EEEEEE", "20px Consolas BOLD"),
            create_text_view_data(f"Goal    : {self._score_to_pass:04d} pt", 705, 260, "#EEEEEE", "20px Consolas BOLD"),
        ]
        scene_progress = create_scene_progress_data(
            frame=self.frame_count, background=backgrounds,
            object_list=game_obj_list,
            foreground=foregrounds, toggle=toggle_objs,
            musics=[BGM01_OBJ],
            sounds=self._sounds

        )
        self._sounds = []

        return scene_progress

    @check_game_result
    def get_game_result(self):
        """
        send game result
        """
        if self.get_game_status() == GameStatus.GAME_PASS:
            self.game_result_state = GameResultState.PASSED
        return {"frame_used": self.frame_count,
                "status": self.game_result_state,
                "attachment": [
                    {
                        "player_num": get_ai_name(0),
                        "rank": 1,
                        "score": self.squid.score,
                        "passed": self.is_passed
                    }
                ]

                }

    def get_keyboard_command(self):
        """
        Define how your game will run by your keyboard
        """
        cmd_1p = []
        key_pressed_list = pygame.key.get_pressed()
        if key_pressed_list[pygame.K_UP]:
            cmd_1p.append("UP")
        elif key_pressed_list[pygame.K_DOWN]:
            cmd_1p.append("DOWN")
        elif key_pressed_list[pygame.K_LEFT]:
            cmd_1p.append("LEFT")
        elif key_pressed_list[pygame.K_RIGHT]:
            cmd_1p.append("RIGHT")
        else:
            cmd_1p.append("NONE")
        return {get_ai_name(0): cmd_1p}

    def _create_foods(self, FOOD_TYPE, count: int = 5):
        for i in range(count):
            # add food to group
            food = FOOD_TYPE(self.foods)
            if isinstance(food, (Food1, Food2, Food3,)):
                # if food pos list is empty , re-create
                if len(self._food_pos_list) < 1:
                    self._food_pos_list = divide_window_into_grid(
                        self._food_window)
                pos = self._food_pos_list.pop()
                food.set_center_x_and_y(
                    pos[0],
                    pos[1]
                )


            elif isinstance(food, (Garbage1, Garbage2, Garbage3,)):
                if len(self._garbage_pos_list) < 1:
                    self._garbage_pos_list = divide_window_into_grid(
                        self._garbage_window, rows=1, cols=10)
                pos = self._garbage_pos_list.pop()
                food.set_center_x_and_y(
                    pos[0],
                    pos[1]
                )

        pass


def divide_window_into_grid(window: WindowConfig, rows: int = 10, cols: int = 10) -> list[(int, int)]:
    grid_positions = []

    # Calculate width and height of each grid piece
    width = (window.right - window.left) // cols
    height = (window.bottom - window.top) // rows

    # Generate grid positions
    for row in range(rows):
        for col in range(cols):
            center_x = window.left + col * width + width // 2
            center_y = window.top + row * height + height // 2
            grid_positions.append((center_x, center_y))

    # Shuffle the list to randomize the order of positions
    random.shuffle(grid_positions)

    return grid_positions
