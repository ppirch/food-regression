import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

VERSION = "baseline"
resnet_50_predict = glob(f"./out/model/foodnet_resnet50_baseline_regression_general_*_677.csv")
test_file = "./regression-20230213T104418Z-001/csv/test.csv"
test_df = pd.read_csv(test_file)
evaluate = []
train_df = pd.read_csv("./regression-20230213T104418Z-001/csv/train.csv")
le = LabelEncoder()
le.fit(train_df["food"])
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

df_error_analysis = test_df.copy()
rsts = []
for rst in resnet_50_predict:
    df = pd.read_csv(rst)
    reg = df["reg"]
    mape = mean_absolute_percentage_error(test_df["weight"], reg)
    # acc = -1
    # if "baseline" not in rst:
    #     classi = df["cls"]
    #     classi = le.inverse_transform(classi)
    #     df_error_analysis["predict_class"] = classi
    #     df_error_analysis["predict_reg"] = reg
    #     df_error_analysis["correct"] = (
    #         df_error_analysis["food"] == df_error_analysis["predict_class"]
    #     )
    #     df_error_analysis["diff"] = (
    #         df_error_analysis["weight"] - df_error_analysis["predict_reg"]
    #     )
    #     df_error_analysis["food_thai"] = df_error_analysis["food"].map(food_mapping)
    #     df_error_analysis["predict_class_thai"] = df_error_analysis[
    #         "predict_class"
    #     ].map(food_mapping)
    #     df_error_analysis["food_thai_to_predict_class_thai"] = (
    #         df_error_analysis["food_thai"]
    #         + " -> "
    #         + df_error_analysis["predict_class_thai"]
    #     )
    #     acc = accuracy_score(classi, test_df["food"])
    curr = {
        "model": rst.split("foodnet_")[-1].split(".")[0],
        "mape": mape,
        "acc": '-',
    }
    rsts.append(curr)
    # rows = []
    # for food in df_error_analysis["food"].unique():
    #     selected_row = df_error_analysis[df_error_analysis["food"] == food]
    #     actual_cls = selected_row["food"]
    #     pred_cls = selected_row["predict_class"]
    #     acture_reg = selected_row["weight"]
    #     pred_reg = selected_row["predict_reg"]
    #     selected_train = train_df[train_df["food"] == food]
    #     rows.append(
    #         {
    #             "food": food,
    #             "food_thai": food_mapping[food],
    #             "mape": mean_absolute_percentage_error(acture_reg, pred_reg),
    #             "min_diff": selected_row["diff"].min(),
    #             "max_diff": selected_row["diff"].max(),
    #             "avg_diff": selected_row["diff"].mean(),
    #             "std_diff": selected_row["diff"].std(),
    #             "acc": accuracy_score(actual_cls, pred_cls),
    #             "train_count": len(selected_train),
    #             "test_count": len(selected_row),
    #             "pct_train": len(selected_train) / len(train_df),
    #             "pct_test": len(selected_row) / len(test_df),
    #             "diff_pct_train_test": len(selected_train) / len(train_df)
    #             - len(selected_row) / len(test_df),
    #             "min_train_weight": selected_train["weight"].min(),
    #             "max_train_weight": selected_train["weight"].max(),
    #             "avg_train_weight": selected_train["weight"].mean(),
    #             "std_train_weight": selected_train["weight"].std(),
    #             "min_test_weight": selected_row["weight"].min(),
    #             "max_test_weight": selected_row["weight"].max(),
    #             "avg_test_weight": selected_row["weight"].mean(),
    #             "std_test_weight": selected_row["weight"].std(),
    #             "diff_avg_weight": (selected_train["weight"].mean()- selected_row["weight"].mean()),
    #         }
    #     )
    # df_error_analysis_by_class = pd.DataFrame(rows)
    # df_error_analysis.to_csv("v2_error_analysis.csv", index=False)
    pd.DataFrame(rsts).to_csv("baseline_general.csv", index=False)
