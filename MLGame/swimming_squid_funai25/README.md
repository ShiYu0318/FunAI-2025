# **Swimming Squid** 魷來魷去

![swimming_squid](https://img.shields.io/github/v/tag/PAIA-Playful-AI-Arena/swimming_squid)

[![MLGame](https://img.shields.io/badge/MLGame->10.4.6a2-<COLOR>.svg)](https://github.com/PAIA-Playful-AI-Arena/MLGame)


這是一個魷魚吃東西小遊戲，茫茫的海洋中有美味的食物，也有人類拋棄的垃圾，請用你的AI幫助小小魷魚平安長大。

![demo](https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid/refs/heads/main/asset/swimming-squid.gif)

---
## 基礎介紹

### 啟動方式

- 直接啟動 [main.py](https://github.com/PAIA-Playful-AI-Arena/swimming_squid/blob/main/main.py) 即可執行

### 遊戲參數設定

```python
# main.py 
game = SwimmingSquid(
            level: int = 1,
            level_file: str = None,
            sound: str = "off")
```
- `level`: 選定內建關卡，預設為 1 選擇第一關。
- `level_file`: 使用外部檔案作為關卡，請注意，使用此設定將會覆蓋掉關卡編號，並且不會自動進入下一關。
- `sound`: 音效。

### 玩法

- 使用鍵盤 上、下、左、右 控制方塊

### 目標

1. 在遊戲時間截止前，盡可能吃到愈多的食物吧！

#### 通關條件

1. 時間結束前，吃到的食物超過`score`，即可過關。

#### 失敗條件

1. 時間結束前，吃到的食物少於`score`，即算失敗。

## 座標系統
1. 使用 pygame 座標系，`左上角`為原點，`X軸`往`右`為正，`Y軸`往`下`為正
2. 回傳的物件座標，皆為物體`中心點`座標


---


## 進階說明

### 使用ＡＩ玩遊戲

```bash
# 在easy game中，打開終端機
python -m mlgame -i ./ml/ml_play_template.py ./ --level 3
python -m mlgame -i ./ml/ml_play_template.py ./ --level_file /path_to_file/level_file.json
```

### ＡＩ範例

```python
import random

class MLPlay:
    def __init__(self,ai_name,*args, **kwargs):
        print("Initial ml script")

    def update(self, scene_info: dict,,*args, **kwargs):

        # print("AI received data from game :", scene_info)

        actions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]

        return random.sample(actions, 1)

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
```

### 遊戲資訊

- scene_info 的資料格式如下

```json
{
  "frame": 15,
  "score": 8,
  "score_to_pass": 10,
  "self_x": 350,
  "self_y": 300,
  "self_h": 60,
  "self_w": 40,
  "self_lv": 1,
  "self_vel": 10,
  "status": "GAME_ALIVE",
  "foods": [
    {
      "h": 30,
      "score": 1,
      "type": "FOOD_1",
      "w": 30,
      "x": 40,
      "y": 134
    },
    {
      "h": 40,
      "score": 2,
      "type": "FOOD_2",
      "w": 40,
      "x": 422,
      "y": 192
    },
    {
      "h": 50,
      "score": 4,
      "type": "FOOD_3",
      "w": 50,
      "x": 264,
      "y": 476
    },
    {
      "h": 30,
      "score": -1,
      "type": "GARBAGE_1",
      "w": 30,
      "x": 100,
      "y": 496
    },
    {
      "h": 40,
      "score": -4,
      "type": "GARBAGE_2",
      "w": 40,
      "x": 633,
      "y": 432
    },
    {
      "h": 50,
      "score": -10,
      "type": "GARBAGE_3",
      "w": 50,
      "x": 54,
      "y": 194
    }
  ],
  "env": {
      "time_to_play": 600,
      "playground_size_w":700, 
      "playground_size_h":550,
      "left": 100,
      "right": 500,
      "top": 100,
      "bottom": 600,
      "food_1": 6,
      "food_2": 4,
      "food_3": 2,
      "garbage_1": 3,
      "garbage_2": 2,
      "garbage_3": 2,
      "score_to_pass": 80
  }

}
```

- `frame`：遊戲畫面更新的編號
- `self_x`：玩家角色的Ｘ座標，表示方塊的`中心點`座標值，單位 pixel。
- `self_y`：玩家角色的Ｙ座標，表示方塊的`中心點`座標值，單位 pixel。
- `self_w`：玩家角色的寬度，單位 pixel。
- `self_h`：玩家角色的高度，單位 pixel。
- `self_vel`：玩家角色的速度，表示方塊每幀移動的像素，單位 pixel。
- `self_lv`：玩家角色的等級，最小 1 ，最大 6。
- `foods`：食物的清單，清單內每一個物件都是一個食物的`中心點`座標值，也會提供此食物是什麼類型和分數多少。
  -  `type` 食物類型： `FOOD_1`, `FOOD_2`, `FOOD_3`, `GARBAGE_1`, `GARBAGE_2`, `GARBAGE_3`
- `score`：目前得到的分數
- `score_to_pass`：通關分數
- `env`：環境資訊，裡面會包含遊戲設定檔的所有參數，也可以拿到邊界資訊。

- `status`： 目前遊戲的狀態
    - `GAME_ALIVE`：遊戲進行中
    - `GAME_PASS`：遊戲通關
    - `GAME_OVER`：遊戲結束

### 動作指令

- 在 update() 最後要回傳一個字串，主角物件即會依照對應的字串行動，一次只能執行一個行動。
    - `UP`：向上移動
    - `DOWN`：向下移動
    - `LEFT`：向左移動
    - `RIGHT`：向右移動
    - `NONE`：原地不動

### 遊戲結果

- 最後結果會顯示在console介面中，若是PAIA伺服器上執行，會回傳下列資訊到平台上。

```json
{
  "frame_used": 100,
  "status": "fail",
  "attachment": [
    {
      "squid": "1P",
      "score": 0,
      "rank": 1,
      "passed": false
    }
  ]
}
```

- `frame_used`：表示使用了多少個frame
- `status`：表示遊戲結束的狀態
  - `passed`:達到指定分數，回傳通過
  - `un_passed`:沒有達到指定分數，回傳不通過
- `attachment`：紀錄遊戲各個玩家的結果與分數等資訊
    - `squid`：玩家編號
    - `score`：吃到的食物總數
    - `rank`：排名
    - `passed`：是否通關

---

## 參考資源
- 音效
    1. https://soundeffect-lab.info/sound/anime/
- 背景音樂
    1. https://www.motionelements.com/zh-hant/stock-music-28190007-bossa-nova-short-loop
- 圖片
    1. 魷魚 https://illustcenter.com/2022/07/03/rdesign_1659/
    2. 湯匙 https://illustcenter.com/2021/11/24/rdesign_6275/
    3. 薯條 https://illustcenter.com/2021/11/16/rdesign_5098/
    4. 空罐 https://illustcenter.com/2021/11/19/rdesign_5772/
    5. 魚1 https://illustcenter.com/2021/12/22/rdesign_8914/
    6. 魚2 https://illustcenter.com/2021/10/28/rdesign_3149/
    7. 蝦子 https://illustcenter.com/2021/10/28/rdesign_3157/