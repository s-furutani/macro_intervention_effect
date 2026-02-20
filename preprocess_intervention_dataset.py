import pandas as pd
import os
import re

def preprocess_pennycook2020():
    directory = "data/intervention_dataset/accuracy_prompts_Pennycook2020"
    path = os.path.join(directory, "raw/Data/Pennycook_et_al__Study_2.csv")
    df_raw = pd.read_csv(path, encoding='mac_roman')
    # print(df_raw.columns.tolist()[:20])
    df_raw = df_raw.rename(columns={"Ô..Condition": "Condition"})
    df = df_raw[df_raw["Finished"] == 1].copy()

    # Fake1_1~Fake1_15 (except RT columns Fake1_RT_...)
    fake_cols = [
        c for c in df.columns
        if c.startswith("Fake1_") and not c.startswith("Fake1_RT_")
    ]

    subset = df[["rid", "Condition"] + fake_cols].copy()

    # wide -> long
    fake_long = subset.melt(
        id_vars=["rid", "Condition"],
        value_vars=fake_cols,
        var_name="item",
        value_name="share"
    )

    fake_long["fake_index"] = (
        fake_long["item"].str.replace("Fake1_", "", regex=False).astype(int)
    )
    fake_long["share01"] = (fake_long["share"] - 1) / 5

    fake_long = fake_long[["rid", "Condition", "fake_index", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    fake_long.to_csv(output_path, index=False)
    print(fake_long.head())


def preprocess_pennycook2021():
    directory = "data/intervention_dataset/accuracy_prompts_Pennycook2021"
    path = os.path.join(directory, "raw/Data and Code/Study_3_data.csv")

    df_raw = pd.read_csv(path)
    df = df_raw.copy()

    df = df[df["Condition"].isin([1, 2])].copy()
    share_cols = [c for c in df.columns if re.fullmatch(r"Fake\d+_3", c)]

    df["id"] = df["confirmCode"].astype(str)

    records = []
    for col in share_cols:
        fake_index = int(col.replace("Fake", "").replace("_3", ""))
        for _, row in df.iterrows():
            value = row[col]
            if pd.isna(value):
                continue
            records.append(
                {
                    "id": row["id"],
                    "Condition": row["Condition"],
                    "fake_index": fake_index,
                    "share": value,
                }
            )

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise RuntimeError("df_long is empty – share columns may be wrong, check patterns.")

    df_long["share01"] = (df_long["share"] - 1) / 5.0

    output_path = os.path.join(directory, "preprocessed_data.csv")
    df_long.to_csv(output_path, index=False)
    print(df_long.head())


def preprocess_fazio2020():
    directory = "data/intervention_dataset/friction_fazio2020"
    path = os.path.join(directory, "raw/share_data_osf.csv")
    df = pd.read_csv(path)
    needed = ["Subject", "Explain"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    df = df[df["Explain"].isin([0, 1])].copy()

    false_cols = [f"S{i}" for i in range(13, 25)]
    for col in false_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found (expected false headline column).")

    share_df = df[["Subject", "Explain"] + false_cols]
    df_long = share_df.melt(
        id_vars=["Subject", "Explain"],
        value_vars=false_cols,
        var_name="item_raw",
        value_name="share"
    ).dropna(subset=["share"])

    df_long["item_id"] = df_long["item_raw"].str.extract(r"S(\d+)").astype(int)
    df_long["item_id"] = df_long["item_id"] - 12  # 13→1, 24→12

    df_long["Condition"] = df_long["Explain"].astype(int)

    df_long = df_long.rename(columns={"Subject": "id"})
    df_long["share01"] = (df_long["share"] - 1) / 5.0

    df_long = df_long[["id", "Condition", "item_id", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    df_long.to_csv(output_path, index=False)
    print(df_long.head())

def preprocess_basol2020():
    directory = "data/intervention_dataset/inoculation_Basol2021"
    path = os.path.join(directory, "raw/Data/Study_2_-_final.xlsx")
    df_raw = pd.read_excel(path, sheet_name="Study 2 - final")

    # 2) 有効なケースだけに絞る（必要に応じて調整）
    df = df_raw.copy()
    df = df[df["Finished"] == True]
    df = df[df["Informed consent"].astype(str).str.startswith("Yes")]

    id_col = "Prolific ID"   # 参加者ID
    cond_col = "Condition"

    def is_fake_sharing_col(col, phase_suffix):
        """
        phase_suffix: "-Pre-Sharing" or "-post-Sharing"
        fake候補: Emotion / Expert / Conspir で始まるもの
        """
        if not col.endswith(phase_suffix):
            return False
        if col.startswith("Emotion-") or col.startswith("Expert-") or col.startswith("Conspir-"):
            return True
        return False

    # Pre と Post それぞれで列名を集める
    pre_suffix = "-Pre-Sharing"
    post_suffix = "-post-Sharing"  # 列名そのままに合わせて小文字/大文字注意

    fake_pre_cols = [c for c in df.columns if is_fake_sharing_col(c, pre_suffix)]
    fake_post_cols = [c for c in df.columns if is_fake_sharing_col(c, post_suffix)]

    # print("Pre fake cols:", fake_pre_cols)
    # print("Post fake cols:", fake_post_cols)

    # 4) wide -> long：pre
    pre_long = df[[id_col, cond_col] + fake_pre_cols].melt(
        id_vars=[id_col, cond_col],
        value_vars=fake_pre_cols,
        var_name="item_raw",
        value_name="share"
    )
    pre_long["phase"] = "pre"

    # item_id（例: Emotion-2-tripl-Pre-Sharing -> Emotion-2-tripl）
    pre_long["item_id"] = pre_long["item_raw"].str.replace(pre_suffix, "", regex=False)

    # 5) wide -> long：post
    post_long = df[[id_col, cond_col] + fake_post_cols].melt(
        id_vars=[id_col, cond_col],
        value_vars=fake_post_cols,
        var_name="item_raw",
        value_name="share"
    )
    post_long["phase"] = "post"
    post_long["item_id"] = post_long["item_raw"].str.replace(post_suffix, "", regex=False)

    # 6) pre/post を結合
    long_all = pd.concat([pre_long, post_long], ignore_index=True)

    # 7) Likert (1〜7 を想定) → [0,1] にスケール
    #    欠損やテキストが紛れていてもいいように数値化してから処理
    long_all["share"] = pd.to_numeric(long_all["share"], errors="coerce")
    long_all = long_all.dropna(subset=["share"])

    # [0,1] スケール: (x-1)/6
    long_all["share01"] = (long_all["share"] - 1) / 6

    # 8) 欲しい列だけに整理
    preprocessed = long_all[[id_col, cond_col, "phase", "item_id", "share", "share01"]].rename(
        columns={
            id_col: "id",
            cond_col: "Condition"
        }
    )

    # 9) CSV に保存
    output_path = os.path.join(directory, "preprocessed_data.csv")
    preprocessed.to_csv(output_path, index=False)

    print(preprocessed.head())

def preprocess_drolsbach2024():
    directory = "data/intervention_dataset/community_notes_Drolsbach2024"
    path = os.path.join(directory, "raw/Data/df_main.csv")
    df_raw = pd.read_csv(path, low_memory=False)
    df = df_raw.copy()
    
    df = df[df["S_FactCheck"].isin(["No Fact-Check", "Community Note"])]
    df = df[df["T_Misleading"] == "Misleading"]    
    df["share01"] = df["S_WillReshare"]
    df["Condition"] = (df["S_FactCheck"] == "Community Note").astype(int)

    preprocessed = df[["P_Id", "T_Id", "Condition", "share01"]].rename(
        columns={
            "P_Id": "id",
            "T_Id": "item_id"
        }
    )

    output_path = os.path.join(directory, "preprocessed_data.csv")
    preprocessed.to_csv(output_path, index=False)
    print(preprocessed.head())
    # print(preprocessed["Condition"].value_counts()) # 0: No Fact-Check, 1: Community Note

# preprocess_pennycook2020()
# preprocess_basol2020()
# preprocess_drolsbach2024()
preprocess_pennycook2021()
# preprocess_fazio2020()