import pandas as pd
VERSION = "result"
eval_df = pd.read_csv(f"./v2_fix_result.csv")
rst = {}
for i, row in eval_df.iterrows():
    model, mape, acc = row
    try:
        name, version, epoch, SEED, A, B = model.split("_")
        epoch, SEED = int(epoch), int(SEED)
        key = f"{name}_{version}_{SEED}_{A}_{B}"
    except:
        try: 
            name, version, epoch, SEED = model.split("_")
            epoch, SEED = int(epoch), int(SEED)
            key = f"{name}_{version}_{SEED}"
        except:
            try: 
                name, version, epoch, SEED, A = model.split("_")
                epoch, SEED = int(epoch), int(SEED)
                key = f"{name}_{version}_{SEED}_{A}"
            except:
                name, version1, version2, epoch, SEED, A = model.split("_")
                epoch, SEED = int(epoch), int(SEED)
                key = f"{name}_{version1}_{version2}_{SEED}_{A}"
    if key not in rst:
        rst[key] = {
            'mape': [-1] * 50,
            'acc': [-1] * 50,
        }
        rst[key]['mape'][epoch-1] = mape
        rst[key]['acc'][epoch-1] = acc
    else:
        rst[key]['mape'][epoch-1] = mape
        rst[key]['acc'][epoch-1] = acc
df_transform_mape= pd.DataFrame()
df_transform_acc = pd.DataFrame()
for key in rst:
    df_transform_mape[key] = rst[key]['mape']
    df_transform_acc[key] = rst[key]['acc']
df_transform_mape.to_csv(f"./out/evaluate_custom_loss_mape_result.csv", index=True)
df_transform_acc.to_csv(f"./out/evaluate_custom_loss_acc_result.csv", index=True)