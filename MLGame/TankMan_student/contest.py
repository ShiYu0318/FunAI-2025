import pygame
from mlgame.game.generic import quit_or_esc
from src.Game import Game
from tqdm import tqdm
import argparse
import importlib



class Contest():
    """
    This class manages a 3 vs 3 game contest.

    Parameters
    ----------
    player : list
        The list of players.
    total_game : int
        The total number of games.
    frame : int
        The frame of the game.
    sound : str
        The sound of the game.
    is_manual : bool
        The manual mode of the game.   

    Returns
    -------
    dict
        The result of the game.      
    """
    def __init__(self, player: list, total_game: int, frame: int, 
                 sound: str, is_manual: bool):
        # initialize player
        self.player1 = player[0]
        self.player2 = player[1]
        self.player3 = player[2]
        self.player4 = player[3]
        self.player5 = player[4]
        self.player6 = player[5]

        # initialize game
        self.total_game = total_game
        self.frame = frame
        self.sound = sound
        self.is_manual = is_manual

        self.user_num = 6
        self.green_team_num = 3
        self.blue_team_num = 3
        
        # initialize game times
        self.game_times = 0
        self.green_team_win = 0
        self.blue_team_win = 0

    def run(self):
        with tqdm(total=self.total_game, unit="round") as pbar:
            pbar.set_description("Playing")
            print()
            for _ in range(self.total_game):
                # initialize game every round
                game = Game(self.user_num, self.green_team_num, 
                            self.blue_team_num, self.is_manual, self.frame, self.sound)
                scene_init_info_dict = game.get_scene_init_data()
                frame_count = 0

                # update game
                while game.is_running() and not quit_or_esc():
                    game.update({
                        "1P": self.player1.update(game.get_data_from_game_to_player()['1P'], []),
                        "2P": self.player2.update(game.get_data_from_game_to_player()['2P'], []),
                        "3P": self.player3.update(game.get_data_from_game_to_player()['3P'], []),
                        "4P": self.player4.update(game.get_data_from_game_to_player()['4P'], []),
                        "5P": self.player5.update(game.get_data_from_game_to_player()['5P'], []),
                        "6P": self.player6.update(game.get_data_from_game_to_player()['6P'], [])                    
                    })                
                    frame_count += 1
                
                # reset player
                self.player1.reset()
                self.player2.reset()
                self.player3.reset()
                self.player4.reset()
                self.player5.reset()
                self.player6.reset()
                self.game_times += 1

                # get game result
                game_result = game.get_game_result()                
                if game_result['attachment'][0]["status"] == "GREEN_TEAM_WIN":
                    self.green_team_win += 1
                else:
                    self.blue_team_win += 1
                
                # update progress bar
                pbar.postfix = f"green_team win: {self.green_team_win} | blue_team win: {self.blue_team_win}"                
                pbar.update(1)
                # print()
                if self.green_team_win > self.total_game/2 or self.blue_team_win > self.total_game/2:
                    break
                
                
        print(f"Game times: {self.game_times} | green_team win: {self.green_team_win} | blue_team win: {self.blue_team_win}")
        pygame.quit()
        return {"green_team_win":self.green_team_win, "blue_team_win":self.blue_team_win}

def import_player(group_number, player_number):
    module_name = f"ml.Group_{group_number}.ml_play_{player_number}"
    module = importlib.import_module(module_name)
    return getattr(module, f"MLPlay")

if __name__ == '__main__':    
    # Set configuration
    sound  = "off"
    total_game  = 9
    frame = 2500
    is_manual = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--green_team", type=str, required=True, help="The folder of the green team")
    parser.add_argument("--blue_team", type=str, required=True, help="The folder of the blue team")
    args = parser.parse_args()
    print(args)
    # player1 = import_player(1, 1)
    
    
    players = [ import_player(args.green_team, 1)('1P', {'sound': sound}),
                import_player(args.green_team, 2)('2P', {'sound': sound}),
                import_player(args.green_team, 3)('3P', {'sound': sound}),
                import_player(args.blue_team, 1)('4P', {'sound': sound}),
                import_player(args.blue_team, 2)('5P', {'sound': sound}),
                import_player(args.blue_team, 3)('6P', {'sound': sound})]
                  
    contest = Contest(players, total_game , frame, sound, is_manual)
    contest.run()
    