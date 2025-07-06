# 手動玩遊戲
python -m mlgame -f 120 -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py . --green_team_num 3 --blue_team_num 3 --is_manual 1 --frame_limit 400

# 1P 使用模型玩遊戲
python -m mlgame -f 120 -i ml/ml_play_model.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py . --green_team_num 3 --blue_team_num 3 --is_manual 1 --frame_limit 400

# 訓練自己的 model
python ml/train.py