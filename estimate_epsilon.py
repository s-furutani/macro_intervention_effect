import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['font.family'] = 'Arial'

def compute_epsilon(df, item_col="item_id", cond_col="Condition", share_col="share01",
                    control_value=None, treat_value=None, s0_min=0.1):
    """
    df: tidy data frame
    item_col: item (a)
    cond_col: condition (0/1 or string)
    control_value: control 群の値
    treat_value: intervention 群の値

    returns: df indexed by item_id with s0, s1, epsilon
    """
    # Control と介入だけに絞る
    df2 = df[df[cond_col].isin([control_value, treat_value])].copy()

    # item × condition で平均 s(i,a) を計算
    tab = (
        df2.groupby([item_col, cond_col])[share_col]
           .mean()
           .unstack()
           .rename(columns={control_value: "s0", treat_value: "s1"})
    )
    # ε(a) = 1 − s1 / s0, s0 >= s0_min
    tab = tab[tab["s0"] >= s0_min] # floor effect removal
    tab["epsilon"] = 1 - tab["s1"] / tab["s0"]

    return tab

def estimate_epsilon_nudge(accuracy_prompt=True):

    if accuracy_prompt:
        # nudge_dir = "data/intervention_dataset/accuracy_prompts_Pennycook2020/"
        nudge_dir = "data/intervention_dataset/accuracy_prompts_Pennycook2021/"
        item_col = "fake_index"
        control_value = 1
        treat_value = 2
    else:
        nudge_dir = "data/intervention_dataset/friction_fazio2020/"
        item_col = "item_id"
        control_value = 0
        treat_value = 1
    nudge_path = nudge_dir + "/preprocessed_data.csv"
    df_nudge = pd.read_csv(nudge_path)
    # Condition: 1=control, 2=intervention
    eps_nudge = compute_epsilon(
        df_nudge,
        item_col=item_col,
        cond_col="Condition",
        share_col="share01",
        control_value=control_value,
        treat_value=treat_value
    )
    return eps_nudge

def estimate_epsilon_prebunk(is_goviral=True):
    if is_goviral:
        treat_val = "GoViral"
    else:
        treat_val = "Infographics"
    prebunk_dir = "data/intervention_dataset/inoculation_Basol2021"
    prebunk_path = prebunk_dir + "/preprocessed_data.csv"
    df_prebunk = pd.read_csv(prebunk_path)
    df_prebunk = df_prebunk[df_prebunk["phase"] == "post"] # post のみ
    eps_prebunk = compute_epsilon(
        df_prebunk,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value="Control",
        treat_value=treat_val
    )
    return eps_prebunk

def estimate_epsilon_contextualization():
    contextualization_dir = "data/intervention_dataset/community_notes_Drolsbach2024"
    contextualization_path = contextualization_dir + "/preprocessed_data.csv"
    df_contextualization = pd.read_csv(contextualization_path)
    # Condition: 0 = No Flag, 1 = Community Note
    eps_contextualization = compute_epsilon(
        df_contextualization,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value=0,
        treat_value=1
    )
    return eps_contextualization

def plot_epsilon_distribution(eps_df, label, bins=15):
    plt.figure(figsize=(7,5))
    sns.histplot(eps_df["epsilon"], bins=bins, kde=True, color="skyblue")
    plt.axvline(eps_df["epsilon"].mean(), color="red", linestyle="--", label=f"mean={eps_df['epsilon'].mean():.3f}")
    plt.title(f"Epsilon Distribution: {label}")
    plt.xlabel("epsilon (relative reduction)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

eps_nudge = estimate_epsilon_nudge(accuracy_prompt=True)
eps_nudge_friction = estimate_epsilon_nudge(accuracy_prompt=False)
eps_prebunk = estimate_epsilon_prebunk(is_goviral=True)
eps_prebunk_infographics = estimate_epsilon_prebunk(is_goviral=False)
eps_contextualization = estimate_epsilon_contextualization()

print("Accuracy Prompt ε(a):")
print(eps_nudge["epsilon"].mean())
print("--------------------------------")
print("Friction Nudge ε(a):")
print(eps_nudge_friction["epsilon"].mean())
print("--------------------------------")
print("Prebunking (Infographics) ε(a):")
print(eps_prebunk_infographics["epsilon"].mean())
print("--------------------------------")
print("Prebunking (GoViral) ε(a):")
print(eps_prebunk["epsilon"].mean())
print("--------------------------------")
print("Contextualization (Community Notes) ε(a):")
print(eps_contextualization["epsilon"].mean())


df_eps_all = pd.concat([
    eps_nudge.reset_index().assign(intervention="Accuracy Prompt"),
    eps_nudge_friction.reset_index().assign(intervention="Friction Nudge"),
    eps_prebunk_infographics.reset_index().assign(intervention="Infographics"),
    eps_prebunk.reset_index().assign(intervention="GoViral"),
    eps_contextualization.reset_index().assign(intervention="Community Notes")
], ignore_index=True)

palette = {
    "Accuracy Prompt": "darkseagreen",
    "Friction Nudge": "seagreen",
    "Infographics": "lightblue",
    "GoViral": "steelblue",
    "Community Notes": "darkorange",
}

plt.figure(figsize=(10,6), dpi=300)
plt.hlines(0, -1, 6, colors="k", linestyles="dashed", linewidth=1)
ax = sns.violinplot(data=df_eps_all, x="intervention", y="epsilon", palette=palette, alpha=0.9)
group_means = df_eps_all.groupby("intervention")["epsilon"].mean()
categories = [t.get_text() for t in ax.get_xticklabels()]
ordered_means = [group_means[c] for c in categories]
for i, mean_val in enumerate(ordered_means):
    plt.text(i+0.3, mean_val, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=20, fontweight="bold", color="black")#, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
plt.xticks(fontsize=15)
plt.yticks(fontsize=16)
plt.xlabel('')
plt.ylabel('Suppression Rate', fontsize=20)
plt.tight_layout()
plt.savefig('results/estimated_epsilon.png')
plt.show()