"""
utils.py — Utility functions for PlayStation Gaming Behavior Analysis
CPSC 222, Spring 2025
Author: Kyle Hays
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Color palette ────────────────────────────────────────────────────────────
PS_BLUE   = "#003087"
PS_SILVER = "#a0a0a0"
PALETTE   = [PS_BLUE, "#0070cc", "#00a6e4", "#e4003a", PS_SILVER]

sns.set_theme(style="whitegrid", palette=PALETTE)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_personal_tracking(path="personal_gaming_tracking.csv"):
    """Load and return the raw personal gaming tracking CSV."""
    df = pd.read_csv(path, parse_dates=["First Played", "Last Played", "Last Update"])
    return df


def load_store_downloads(path="playstation_store_top_downloads.csv"):
    """Load and return the raw PlayStation Store top downloads CSV."""
    df = pd.read_csv(path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_personal_tracking(df):
    """
    Clean the personal tracking DataFrame:
      - Remove streaming/media apps (non-games)
      - Drop rows with missing Hours Played
      - Compute Days Active and Avg Session Length
      - Assign genre labels
      - Label platform generation
    Returns a cleaned copy.
    """
    non_games = {"Crunchyroll", "Netflix", "Twitch", "YouTube", "Max",
                 "HBO Max", "Disney+", "ESPN", "Prime Video", "Spotify",
                 "Peacock", "Funimation"}
    cleaned = df[~df["Game"].isin(non_games)].copy()
    cleaned = cleaned.dropna(subset=["Hours Played"])
    cleaned = cleaned[cleaned["Hours Played"] > 0].copy()

    # Duration of active engagement (days)
    cleaned["Days Active"] = (
        (cleaned["Last Played"] - cleaned["First Played"]).dt.days + 1
    )

    # Average session length (hours per session)
    cleaned["Avg Session Hr"] = (
        cleaned["Hours Played"] / cleaned["Sessions"]
    ).round(2)

    # Engagement tier (class label)
    bins   = [0, 10, 50, 150, np.inf]
    labels = ["Casual", "Moderate", "Engaged", "Hardcore"]
    cleaned["Engagement Tier"] = pd.cut(
        cleaned["Hours Played"], bins=bins, labels=labels
    )

    # Broad genre mapping
    genre_map = {
        "Genshin Impact": "Gacha/RPG", "Honkai: Star Rail": "Gacha/RPG",
        "Wuthering Waves: Version 2.0": "Gacha/RPG",
        "Zenless Zone Zero": "Gacha/RPG",
        "NBA 2K23": "Sports", "NBA 2K24": "Sports", "NBA 2K22": "Sports",
        "NBA 2K26": "Sports", "NBA 2K25": "Sports", "NBA 2K21": "Sports",
        "NBA 2K17": "Sports", "NBA 2K16": "Sports", "NBA 2K20": "Sports",
        "NBA LIVE 19": "Sports",
        "MLB The Show 25": "Sports", "MLB The Show 26": "Sports",
        "MLB The Show 23": "Sports", "MLB The Show 22": "Sports",
        "MLB The Show 21": "Sports", "MLB The Show 17": "Sports",
        "EA SPORTS College Football 26": "Sports", "FIFA 22": "Sports",
        "Baldur's Gate 3": "RPG", "Persona 5": "RPG",
        "Persona 5 Royal": "RPG", "Persona 3 Reload": "RPG",
        "Metaphor: ReFantazio": "RPG", "Cyberpunk 2077": "Action/RPG",
        "ELDEN RING": "Action/RPG", "Diablo IV": "Action/RPG",
        "The Witcher 3: Wild Hunt": "Action/RPG",
        "Like a Dragon: Infinite Wealth PS4 & PS5": "RPG",
        "FINAL FANTASY VII REBIRTH": "RPG", "FINAL FANTASY VII REMAKE": "RPG",
        "FINAL FANTASY XVI": "RPG", "FINAL FANTASY XV": "RPG",
        "FINAL FANTASY VII": "RPG", "OCTOPATH TRAVELER II": "RPG",
        "Persona 5 Strikers": "Action/RPG",
        "Marvel Rivals": "Action/Shooter",
        "Call of Duty: Modern Warfare II": "Action/Shooter",
        "Destiny 2": "Action/Shooter", "Apex Legends": "Action/Shooter",
        "Overwatch 2": "Action/Shooter",
        "Ghost of Tsushima": "Action/Adventure",
        "God of War Ragnarok": "Action/Adventure",
        "Marvel's Spider-Man 2": "Action/Adventure",
        "Rise of the Ronin": "Action/Adventure",
        "Ghost of Yotei": "Action/Adventure",
        "Assassin's Creed Shadows": "Action/Adventure",
        "Hogwarts Legacy": "Action/Adventure",
        "STAR WARS Jedi: Survivor": "Action/Adventure",
        "Cyberpunk 2077": "Action/RPG",
        "Clair Obscur: Expedition 33": "RPG",
        "Hollow Knight: Silksong": "Indie/Platformer",
        "Hollow Knight": "Indie/Platformer",
        "Sea of Stars": "Indie/Platformer",
        "Spiritfarer": "Indie/Platformer",
        "Hades": "Indie/Platformer",
        "LEGO Star Wars: The Skywalker Saga": "Family/Party",
        "Minecraft": "Sandbox",
        "Granblue Fantasy: Relink": "Action/RPG",
    }
    cleaned["Genre"] = cleaned["Game"].map(genre_map).fillna("Other")

    # Platform generation flag
    cleaned["Gen"] = cleaned["Platform"].map({"PS5": "PS5", "PS4": "PS4"})

    return cleaned


def clean_store_downloads(df):
    """
    Clean the store downloads DataFrame:
      - Normalize game names (strip extra spaces)
      - Deduplicate by game+year keeping US/Canada first
    Returns a cleaned copy.
    """
    cleaned = df.copy()
    cleaned["Game"] = cleaned["Game"].str.strip()
    # Keep US/Canada record first when de-duping by game+year
    priority = {"US/Canada": 0, "EU": 1}
    cleaned["_priority"] = cleaned["Region"].map(priority)
    cleaned = (
        cleaned.sort_values("_priority")
               .drop_duplicates(subset=["Game", "Year", "Category"])
               .drop(columns="_priority")
               .reset_index(drop=True)
    )
    return cleaned


def merge_datasets(personal_df, store_df):
    """
    Merge personal tracking with store download data on game name.
    Uses a fuzzy-enough join: strip and lower-case both sides.
    Returns merged DataFrame.
    """
    p = personal_df.copy()
    s = store_df.copy()
    p["_key"] = p["Game"].str.strip().str.lower()
    s["_key"] = s["Game"].str.strip().str.lower()

    # Aggregate store data: was this game ever a top download, and in which year?
    s_agg = (
        s.groupby("_key")
         .agg(Top_Download_Year=("Year", "min"),
              Store_Category=("Category", "first"),
              Store_Region=("Region", "first"))
         .reset_index()
    )
    s_agg["Was_Top_Download"] = True

    merged = p.merge(s_agg, on="_key", how="left")
    merged["Was_Top_Download"] = merged["Was_Top_Download"].fillna(False)
    merged = merged.drop(columns="_key")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def summary_stats(df, numeric_cols=None):
    """Return descriptive statistics for key numeric columns."""
    if numeric_cols is None:
        numeric_cols = ["Hours Played", "Sessions", "Avg Session Hr", "Days Active"]
    return df[numeric_cols].describe().round(2)


def plot_top_games(df, n=15, title="Top Games by Hours Played"):
    """Bar chart of top N games by hours played."""
    top = df.nlargest(n, "Hours Played")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["Game"][::-1], top["Hours Played"][::-1], color=PS_BLUE)
    ax.bar_label(bars, fmt="%.0f hrs", padding=4, fontsize=8)
    ax.set_xlabel("Hours Played")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_hours_distribution(df):
    """Histogram + KDE of hours played (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    log_vals = np.log1p(df["Hours Played"])
    ax.hist(log_vals, bins=25, color=PS_BLUE, edgecolor="white", alpha=0.85)
    ax.set_xlabel("log(1 + Hours Played)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Hours Played (log scale)", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_genre_breakdown(df, col="Genre", value="Hours Played"):
    """Pie chart of hours by genre."""
    grouped = df.groupby(col)[value].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        grouped, labels=grouped.index, autopct="%1.1f%%",
        startangle=140, colors=sns.color_palette(PALETTE, len(grouped))
    )
    ax.set_title(f"Total {value} by {col}", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_platform_comparison(df):
    """Box plot comparing hours played on PS4 vs PS5."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Gen", y="Hours Played", palette=PALETTE, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Platform Generation")
    ax.set_ylabel("Hours Played (log scale)")
    ax.set_title("Hours Played: PS4 vs PS5", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_engagement_tier_distribution(df):
    """Count plot of engagement tiers."""
    order = ["Casual", "Moderate", "Engaged", "Hardcore"]
    fig, ax = plt.subplots(figsize=(7, 4))
    tier_counts = df["Engagement Tier"].value_counts().reindex(order)
    ax.bar(tier_counts.index, tier_counts.values, color=PALETTE[:4], edgecolor="white")
    for i, v in enumerate(tier_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
    ax.set_xlabel("Engagement Tier")
    ax.set_ylabel("Number of Games")
    ax.set_title("Game Count by Engagement Tier", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_sessions_vs_hours(df):
    """Scatter plot of Sessions vs Hours Played, colored by Engagement Tier."""
    tier_colors = {"Casual": "#a0a0a0", "Moderate": "#0070cc",
                   "Engaged": "#00a6e4", "Hardcore": "#003087"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier, grp in df.groupby("Engagement Tier"):
        ax.scatter(grp["Sessions"], grp["Hours Played"],
                   label=tier, alpha=0.75, s=40,
                   color=tier_colors.get(tier, "grey"))
    ax.set_xlabel("Sessions")
    ax.set_ylabel("Hours Played")
    ax.set_title("Sessions vs Hours Played by Engagement Tier", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_yearly_activity(df):
    """Bar chart of hours played per calendar year (based on First Played year)."""
    df2 = df.copy()
    df2["Year"] = df2["First Played"].dt.year
    yearly = df2.groupby("Year")["Hours Played"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index, yearly.values, color=PS_BLUE, edgecolor="white")
    ax.set_xlabel("Year First Played")
    ax.set_ylabel("Total Hours Played")
    ax.set_title("Total Hours Played by Year First Engaged", fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def plot_store_rank_heatmap(store_df):
    """Heatmap of which games appeared in yearly top charts by category."""
    pivot = store_df.pivot_table(
        index="Game", columns="Year", values="Rank", aggfunc="min"
    )
    pivot = pivot.dropna(thresh=2).sort_index()
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.25)))
    sns.heatmap(pivot, cmap="Blues_r", linewidths=0.3,
                annot=True, fmt=".0f", ax=ax, cbar_kws={"label": "Rank"})
    ax.set_title("PlayStation Store Top-Download Rank by Year", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Correlation heatmap for numeric features."""
    num_cols = ["Hours Played", "Sessions", "Avg Session Hr", "Days Active"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix — Numeric Features", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_top_store_games(store_df, top_n=20):
    """Bar chart of most frequently charted games in store data."""
    freq = store_df["Game"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(freq.index[::-1], freq.values[::-1], color=PS_BLUE)
    ax.set_xlabel("Times Appeared in Top-Download Charts")
    ax.set_title(f"Top {top_n} Most Frequently Charted PS Store Games (2023-2025)",
                 fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTING
# ══════════════════════════════════════════════════════════════════════════════

def t_test_platform_hours(df):
    """
    Independent-samples t-test: PS5 hours vs PS4 hours.
    Returns (t_stat, p_value).
    """
    from scipy import stats
    ps5 = df[df["Gen"] == "PS5"]["Hours Played"].dropna()
    ps4 = df[df["Gen"] == "PS4"]["Hours Played"].dropna()
    t, p = stats.ttest_ind(ps5, ps4, equal_var=False)
    return round(t, 4), round(p, 4)


def t_test_top_download_hours(df):
    """
    Independent-samples t-test: hours for top-downloaded games vs others.
    Returns (t_stat, p_value).
    """
    from scipy import stats
    top   = df[df["Was_Top_Download"] == True]["Hours Played"].dropna()
    other = df[df["Was_Top_Download"] == False]["Hours Played"].dropna()
    t, p  = stats.ttest_ind(top, other, equal_var=False)
    return round(t, 4), round(p, 4)


def chi_square_genre_tier(df):
    """
    Chi-square test of independence: Genre vs Engagement Tier.
    Returns (chi2, p_value, degrees_of_freedom).
    """
    from scipy import stats
    ct = pd.crosstab(df["Genre"], df["Engagement Tier"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return round(chi2, 4), round(p, 4), dof


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df, target="Engagement Tier"):
    """
    Build feature matrix X and label vector y for classification.
    Features: Sessions, Avg Session Hr, Days Active, Gen (encoded), Genre (encoded).
    Returns X (DataFrame), y (Series), feature_names (list).
    """
    data = df.dropna(subset=[target, "Sessions", "Avg Session Hr",
                              "Days Active", "Gen", "Genre"]).copy()

    le_gen   = LabelEncoder()
    le_genre = LabelEncoder()
    data["Gen_enc"]   = le_gen.fit_transform(data["Gen"])
    data["Genre_enc"] = le_genre.fit_transform(data["Genre"])

    feature_names = ["Sessions", "Avg Session Hr", "Days Active",
                     "Gen_enc", "Genre_enc"]
    X = data[feature_names]
    y = data[target].astype(str)
    return X, y, feature_names


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Train/test split + StandardScaler. Returns X_train, X_test, y_train, y_test, scaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)
    return X_train_s, X_test_s, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — kNN
# ══════════════════════════════════════════════════════════════════════════════

def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """Cross-validate kNN for a range of k values. Returns (best_k, scores_dict)."""
    scores = {}
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv  = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        scores[k] = cv.mean()
    best_k = max(scores, key=scores.get)
    return best_k, scores


def plot_k_vs_accuracy(scores_dict):
    """Line plot of k vs cross-val accuracy."""
    ks  = list(scores_dict.keys())
    acc = list(scores_dict.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, acc, marker="o", color=PS_BLUE)
    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("kNN: k vs Cross-Validation Accuracy", fontweight="bold")
    best_k = max(scores_dict, key=scores_dict.get)
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax.legend()
    plt.tight_layout()
    return fig


def train_knn(X_train, y_train, k):
    """Train and return a kNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def evaluate_classifier(clf, X_test, y_test, title="Classifier"):
    """Print classification report and return (accuracy, conf_matrix_figure)."""
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {title} — Test Accuracy: {acc:.4f}")
    print('='*50)
    print(classification_report(y_test, y_pred, zero_division=0))

    cm  = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{title} — Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    return acc, fig


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — Decision Tree
# ══════════════════════════════════════════════════════════════════════════════

def find_best_depth(X_train, y_train, depth_range=range(1, 11)):
    """Cross-validate decision tree for max_depth values. Returns (best_depth, scores_dict)."""
    scores = {}
    for d in depth_range:
        dt  = DecisionTreeClassifier(max_depth=d, random_state=42)
        cv  = cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")
        scores[d] = cv.mean()
    best_depth = max(scores, key=scores.get)
    return best_depth, scores


def plot_depth_vs_accuracy(scores_dict):
    """Line plot of max_depth vs cross-val accuracy."""
    ds  = list(scores_dict.keys())
    acc = list(scores_dict.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ds, acc, marker="s", color="#e4003a")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("Decision Tree: max_depth vs Cross-Validation Accuracy",
                 fontweight="bold")
    best_d = max(scores_dict, key=scores_dict.get)
    ax.axvline(best_d, color="blue", linestyle="--", label=f"Best depth={best_d}")
    ax.legend()
    plt.tight_layout()
    return fig


def train_decision_tree(X_train, y_train, max_depth):
    """Train and return a DecisionTreeClassifier."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt


def plot_decision_tree(dt, feature_names, class_names):
    """Visualize the decision tree."""
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(dt, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Decision Tree Visualization", fontweight="bold")
    plt.tight_layout()
    return fig


def feature_importance_plot(dt, feature_names):
    """Horizontal bar chart of feature importances."""
    importances = pd.Series(dt.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color=PS_BLUE)
    ax.set_xlabel("Importance")
    ax.set_title("Decision Tree — Feature Importances", fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT / SAVING
# ══════════════════════════════════════════════════════════════════════════════

def save_cleaned_csv(df, path):
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def print_missing_summary(df):
    """Print a summary of missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values:\n", missing.to_string())
