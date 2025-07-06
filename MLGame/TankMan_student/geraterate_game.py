import importlib
from contest import Contest
def import_player(group_number, player_number):
    module_name = f"ml.Group_{group_number}.ml_play_{player_number}"
    module = importlib.import_module(module_name)
    return getattr(module, f"MLPlay")

num_groups = 3
players_per_group = 3

players = {}


Group_mapping = {
    "A" : "1",
    "B" : "1",
    "C" : "1",
    "D" : "1",
    "E" : "1",
    "F" : "1",
    "G" : "1",
    "H" : "1",
    "I" : "1",
    }


for group in Group_mapping:    
    for player in range(1, players_per_group + 1):
        player_key = f"player{group}_{player}"        
        players[player_key] = import_player(Group_mapping[group], player)

record = {}
for group in Group_mapping:
    record[f"Group_{group}"] = 0

if __name__ == '__main__':
    sound = "off"
    total_game = 5
    frame = 2500
    is_manual = False
        
    
    Games = []
    for group_i in Group_mapping:
        for group_j in Group_mapping:                  
            if group_i != group_j:
                game = { "players" : [
                            players[f"player{group_i}_{k+1}"] for k in range(players_per_group)] 
                            + [players[f"player{group_j}_{k+1}"] for k in range(players_per_group)],
                         "home": group_i,
                          "away": group_j
                        }                                
                Games.append(game)
    
    
    for game in Games:            
        
        selected_players = [game["players"][0]('1P', {'sound': sound}),
                            game["players"][1]('2P', {'sound': sound}),
                            game["players"][2]('3P', {'sound': sound}),
                            game["players"][3]('4P', {'sound': sound}),
                            game["players"][4]('5P', {'sound': sound}),
                            game["players"][5]('6P', {'sound': sound})]

        contest_instance = Contest(selected_players, total_game, frame, sound, is_manual)
        result = contest_instance.run()
      
        if result["green_team_win"] > result["blue_team_win"]:
            record[f"Group_{game['home']}"] += 1
        else:
            record[f"Group_{game['away']}"] += 1
        print("home:", "Group_"+str(game["home"]), result["green_team_win"], "VS", result["blue_team_win"] ,"away", "Group_"+str(game["away"]))
        print(record)
        input("Press Enter to continue...")


