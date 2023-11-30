import pandas as pd
import os
import shutil

df = pd.read_csv("./img.csv")
train_df = pd.read_csv("./regression/csv/train.csv")
test_df = pd.read_csv('./regression/csv/test.csv')

path = './regression/test/'

img_food = {}
food_mapping = {
    "food1": "แกงขนุนหมู",
    "food2": "แกงเขียวหวานไก่",
    "food3": "แกงเขียวหวานหมู",
    "food4": "แกงจืดกะหล่ำปลีหมู",
    "food5": "แกงจืดเต้าหู้ไข่",
    "food6": "แกงจืดแตงกวาหมู",
    "food7": "แกงจืดผักกาดขาวหมู",
    "food8": "แกงจืดวุ้นเส้น",
    "food9": "แกงจืดหัวไชเท้าหมู",
    "food10": "แกงผักกาดเขียวไก่",
    "food11": "แกงเผ็ดหมู",
    "food12": "แกงฟักเขียวไก่",
    "food13": "แกงส้มผักบุ้งหมู",
    "food14": "แกงอ่อมหมู",
    "food15": "แกงฮังเลหมู",
    "food16": "ไก่ผัดพริกสด",
    "food17": "ไก่อบ",
    "food18": "ข้าวสวย",
    "food19": "ไข่พะโล้",
    "food20": "จอผักกาด",
    "food21": "ต้มข่าไก่",
    "food22": "ต้มส้มขาหมูเห็ด",
    "food23": "ตุ๋นฟักเขียวไก่",
    "food24": "น้ำพริกอ่อง",
    "food25": "ผัดกะเพราไก่บดถั่วฝักยาว",
    "food26": "ผัดกะเพราหมู",
    "food27": "ผัดกะหล่ำปลีเต้าหู้",
    "food28": "ผัดซีอิ๊ว",
    "food29": "ผัดถั่วงอกเต้าหู้เหลือง",
    "food30": "ผัดผักกาดขาวหมูบด",
    "food31": "ผัดผักรวมหมู",
    "food32": "ผัดผักกะหล่ำปลีฝอยหมู",
    "food33": "ผัดวุ้นเส้น",
    "food34": "ผัดหมูโบราณ",
    "food35": "พะแนงหมู",
    "food36": "ยำไก่เมืองหัวปลี",
    "food37": "ลาบหมูเมือง ",
    "food38": "หมูผัดพริกขิง",
    "food39": "หลนปลาเค็ม",
    "food40": "หุ้มไก่",
    "food41": "ข้าวกล้อง",
    "food42": "โจ๊ก",
    "food43": "หมูผัดพริกแกง",
}

for index, row in test_df.iterrows():
    filename = row['filename']
    food = row['food']
    food_thai = food_mapping[food]
    img_food[filename] = f'{food}_{food_thai}/'

for index, row in df.iterrows():
    img = row['filename']
    src = os.path.join(path, img)
    if not os.path.exists('./img_analysis/' + img_food[img]):
        os.mkdir('./img_analysis/' + img_food[img])
    dst = os.path.join('./img_analysis/' + img_food[img], img)
    shutil.copy(src, dst)

# sample 10 row from train_df based on food
for food in train_df['food'].unique():
    sample_df = train_df[train_df['food'] == food].sample(10)
    for index, row in sample_df.iterrows():
        img = row['filename']
        food = row['food']
        food_thai = food_mapping[food]
        key = f'{food}_{food_thai}/'
        src = os.path.join('./regression/train/', img)
        if not os.path.exists('./img_analysis/' + key):
            os.mkdir('./img_analysis/' + key)
        dst = os.path.join('./img_analysis/' + key, img)
        shutil.copy(src, dst)