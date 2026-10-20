from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from xgboost import XGBClassifier

from train_models import MODELS_DIR, REPORTS_DIR, load_processed


CHARTS_DIR = REPORTS_DIR / "charts"
RANDOM_STATE = 42

SCORE_FEATURES = ["runs_before", "wickets_before", "legal_balls_before"]
CHASE_FEATURES = ["required_runs_before", "remaining_balls_before", "wickets_in_hand_before", "required_rr_before"]
CHASE_FEATURE_LABELS = {
    "required_runs_before": "Runs needed",
    "remaining_balls_before": "Balls left",
    "wickets_in_hand_before": "Wickets in hand",
    "required_rr_before": "Required run rate",
}

COACH_ACTIONS = ["Defend", "Rotate", "Attack"]
COACH_OUTCOMES = ["0", "1", "2", "4", "6", "W"]
# The chase state is grouped so a table-based learner can cover a full 120-ball chase:
# balls left by over (20 groups), required run rate (9 groups) and wickets in hand (5 groups).
COACH_BALLS_EDGES = list(range(6, 120, 6))
COACH_RATE_EDGES = [6.0, 7.5, 9.0, 10.5, 12.0, 14.0, 17.0, 21.0]
COACH_WICKETS_EDGES = [2, 4, 6, 8]
COACH_PHASES = ["powerplay", "middle", "death"]
COACH_RUNS_FOR_OUTCOME = [0, 1, 2, 4, 6, 0]


def edge_labels(edges: list, lowest: float, highest: float, step: float = 1) -> list[str]:
    labels, start = [], lowest
    for edge in edges:
        labels.append(f"{start:g}-{edge:g}")
        start = edge + step
    labels.append(f"{start:g}-{highest:g}" if highest != float("inf") else f"{start:g}+")
    return labels


COACH_BALLS_BUCKETS = edge_labels(COACH_BALLS_EDGES, 1, 120)
COACH_RATE_BUCKETS = ["up to 6"] + [f"{low:g}-{high:g}" for low, high in zip(COACH_RATE_EDGES, COACH_RATE_EDGES[1:])] + ["21+"]
COACH_WICKETS_BUCKETS = edge_labels(COACH_WICKETS_EDGES, 1, 10)


def holdout_split(seasons: pd.Series) -> tuple[pd.Series, list[int]]:
    holdouts = sorted(seasons.unique().tolist())[-3:]
    return seasons.isin(holdouts), holdouts


def build_eda_charts(deliveries: pd.DataFrame, matches: pd.DataFrame) -> list[dict]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    # Dark styling so the PNGs sit naturally inside the app's dark cards.
    sns.set_theme(
        style="darkgrid",
        palette=["#f6b94d", "#1ec8a5", "#5cbcff"],
        rc={
            "figure.facecolor": "#0b192c",
            "axes.facecolor": "#10223a",
            "axes.edgecolor": "#20314e",
            "grid.color": "#20314e",
            "text.color": "#f4f7fb",
            "axes.labelcolor": "#b5c2d7",
            "xtick.color": "#b5c2d7",
            "ytick.color": "#b5c2d7",
        },
    )
    charts: list[dict] = []

    def save(name: str, title: str, description: str) -> None:
        plt.title(title)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"{name}.png", dpi=110)
        plt.close()
        charts.append({"name": name, "title": title, "description": description})

    overs = deliveries.groupby(["match_id", "innings", "over", "phase"], as_index=False)["runs_total"].sum()
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=overs, x="phase", y="runs_total", order=["powerplay", "middle", "death"], hue="phase", legend=False, linecolor="#f4f7fb", fliersize=2)
    plt.xlabel("Phase")
    plt.ylabel("Runs in the over")
    save("runs_by_phase", "Runs per over by phase", "Death overs have the highest median and the widest spread of runs.")

    decided = matches[matches["winner"] != ""]
    toss = decided.groupby("toss_decision", as_index=False)["toss_winner_won"].mean()
    toss["toss_winner_won"] *= 100
    plt.figure(figsize=(6, 4))
    sns.barplot(data=toss, x="toss_decision", y="toss_winner_won", hue="toss_decision", legend=False)
    plt.axhline(50, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Toss decision")
    plt.ylabel("Toss winner won (%)")
    save("toss_decision_wins", "Does the toss decision matter?", "Win rate of the toss winner after choosing to bat or field.")

    chase = deliveries[(deliveries["innings"] == 2) & (deliveries["winner"] != "")]
    sample = chase.sample(n=min(20000, len(chase)), random_state=RANDOM_STATE)
    columns = ["runs_before", "wickets_before", "legal_balls_before", "current_rr_before", "required_runs_before", "required_rr_before", "batting_team_won"]
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(sample[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False)
    save("feature_correlation", "Chase feature correlation", "How match-state features relate to each other and to the chasing team winning.")

    trend = matches.groupby("season_year", as_index=False)["first_innings_runs"].mean()
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=trend, x="season_year", y="first_innings_runs", marker="o")
    plt.xlabel("Season")
    plt.ylabel("Average first-innings total")
    save("season_scoring_trend", "First-innings scoring by season", "Average first-innings total across IPL seasons.")
    return charts


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
    }


def train_score_models(deliveries: pd.DataFrame) -> dict:
    first = deliveries[(deliveries["innings"] == 1) & (deliveries["legal_balls_before"] < 120)].dropna(subset=["first_innings_runs"])
    test_mask, holdouts = holdout_split(first["season_year"])

    snapshot = first[first["legal_balls_before"] == 60].drop_duplicates("match_id")
    snapshot_test = snapshot["season_year"].isin(holdouts)
    simple = LinearRegression().fit(snapshot.loc[~snapshot_test, ["runs_before"]], snapshot.loc[~snapshot_test, "first_innings_runs"])
    simple_report = {
        "intercept": round(float(simple.intercept_), 2),
        "slope": round(float(simple.coef_[0]), 3),
        **regression_metrics(snapshot.loc[snapshot_test, "first_innings_runs"], simple.predict(snapshot.loc[snapshot_test, ["runs_before"]])),
    }

    train = first[~test_mask].sample(n=min(30000, int((~test_mask).sum())), random_state=RANDOM_STATE)
    test = first[test_mask].sample(n=min(10000, int(test_mask.sum())), random_state=RANDOM_STATE)
    multiple = LinearRegression().fit(train[SCORE_FEATURES], train["first_innings_runs"])
    multiple_report = {
        "intercept": round(float(multiple.intercept_), 2),
        "coefficients": {name: round(float(value), 3) for name, value in zip(SCORE_FEATURES, multiple.coef_)},
        **regression_metrics(test["first_innings_runs"], multiple.predict(test[SCORE_FEATURES])),
    }

    svr_train = train.sample(n=min(5000, len(train)), random_state=RANDOM_STATE)
    svr = make_pipeline(StandardScaler(), SVR(C=50, epsilon=2.0)).fit(svr_train[SCORE_FEATURES], svr_train["first_innings_runs"])
    svr_report = regression_metrics(test["first_innings_runs"], svr.predict(test[SCORE_FEATURES]))

    joblib.dump(
        {"simple": simple, "multiple": multiple, "svr": svr, "features": SCORE_FEATURES},
        MODELS_DIR / "score_models.joblib",
    )
    return {"holdout_seasons": holdouts, "simple": simple_report, "multiple": multiple_report, "svr": svr_report}


def readable_node_text(text: str) -> str:
    """Turn plot_tree's raw node text into plain cricket language."""
    lines = text.split("\n")
    rule = next((line for line in lines if "<=" in line), None)
    share = next(line for line in lines if line.startswith("samples")).split("=")[1].strip()
    values = next(line for line in lines if line.startswith("value")).split("=")[1].strip(" []").split(",")
    chase_win = float(values[1]) * 100
    summary = f"Chasing side won {chase_win:.0f}%\n{share} of match situations"
    if rule:
        return f"{rule.replace('<=', '≤')}?\n{summary}"
    return f"Prediction: {'chase WINS' if chase_win >= 50 else 'chase LOSES'}\n{summary}"


def plot_decision_tree(tree: DecisionTreeClassifier) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 8))
    figure.patch.set_facecolor("#0b192c")
    nodes = plot_tree(
        tree,
        feature_names=[CHASE_FEATURE_LABELS[name] for name in CHASE_FEATURES],
        class_names=["Defending side wins", "Chasing side wins"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=11,
        ax=axis,
    )
    for node in nodes:
        if "samples" in node.get_text():
            node.set_text(readable_node_text(node.get_text()))
        else:
            node.set_color("#f4f7fb")
        if node.arrow_patch is not None:
            node.arrow_patch.set_color("#b5c2d7")
    axis.set_title("Decision tree for chases (left branch = condition true, right branch = false)", color="#f4f7fb", fontsize=14)
    figure.tight_layout()
    figure.savefig(CHARTS_DIR / "decision_tree.png", dpi=110)
    plt.close(figure)
    return "decision_tree"


def classification_metrics(model, features: pd.DataFrame, target: pd.Series) -> dict:
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(target, probabilities >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
    }


def train_chase_models(deliveries: pd.DataFrame) -> dict:
    chase = deliveries[
        (deliveries["innings"] == 2)
        & (deliveries["winner"] != "")
        & (deliveries["remaining_balls_before"] > 0)
        & (deliveries["required_runs_before"] > 0)
    ].dropna(subset=CHASE_FEATURES)
    test_mask, holdouts = holdout_split(chase["season_year"])
    train = chase[~test_mask].sample(n=min(40000, int((~test_mask).sum())), random_state=RANDOM_STATE)
    test = chase[test_mask].sample(n=min(20000, int(test_mask.sum())), random_state=RANDOM_STATE)
    x_train, y_train = train[CHASE_FEATURES], train["batting_team_won"].astype(int)
    x_test, y_test = test[CHASE_FEATURES], test["batting_team_won"].astype(int)
    svm_train = train.sample(n=8000, random_state=RANDOM_STATE)

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(x_train, y_train)
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE).fit(x_train, y_train)
    svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)).fit(
        svm_train[CHASE_FEATURES], svm_train["batting_team_won"].astype(int)
    )
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(x_train, y_train)
    base_xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss").fit(x_train, y_train)

    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={"max_depth": [2, 3], "min_samples_leaf": [50, 200, 500, 1000]},
        scoring="roc_auc",
        cv=3,
    ).fit(x_train, y_train)
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        },
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
    ).fit(x_train.iloc[:20000], y_train.iloc[:20000])
    tree, xgb = tree_search.best_estimator_, xgb_search.best_estimator_

    models = {
        "Logistic Regression": logistic,
        "Decision Tree": tree,
        "SVM": svm,
        "AdaBoost": adaboost,
        "XGBoost": xgb,
    }
    leaderboard = sorted(
        ({"model": name, **classification_metrics(model, x_test, y_test)} for name, model in models.items()),
        key=lambda row: row["roc_auc"],
        reverse=True,
    )
    tuning = [
        {
            "model": "Decision Tree",
            "method": "GridSearchCV",
            "best_params": tree_search.best_params_,
            "before": classification_metrics(base_tree, x_test, y_test),
            "after": classification_metrics(tree, x_test, y_test),
        },
        {
            "model": "XGBoost",
            "method": "RandomizedSearchCV",
            "best_params": {key: (float(value) if isinstance(value, float) else int(value)) for key, value in xgb_search.best_params_.items()},
            "before": classification_metrics(base_xgb, x_test, y_test),
            "after": classification_metrics(xgb, x_test, y_test),
        },
    ]

    joblib.dump(
        {"models": models, "features": CHASE_FEATURES, "feature_labels": CHASE_FEATURE_LABELS},
        MODELS_DIR / "chase_models.joblib",
    )
    return {
        "holdout_seasons": holdouts,
        "features": [CHASE_FEATURE_LABELS[name] for name in CHASE_FEATURES],
        "leaderboard": leaderboard,
        "tuning": tuning,
        "tree_chart": plot_decision_tree(tree),
        "tree_rules": export_text(tree, feature_names=[CHASE_FEATURE_LABELS[name] for name in CHASE_FEATURES], decimals=1),
    }


def detect_freak_innings(deliveries: pd.DataFrame) -> list[dict]:
    legal = deliveries["legal_delivery"] == 1
    frame = deliveries.assign(
        legal_ball=legal.astype(int),
        four=(deliveries["runs_batter"] == 4).astype(int),
        six=(deliveries["runs_batter"] == 6).astype(int),
        dot=(legal & (deliveries["runs_total"] == 0)).astype(int),
    )
    innings = (
        frame.groupby(["match_id", "innings"], as_index=False)
        .agg(
            batting_team=("batting_team", "first"),
            fielding_team=("fielding_team", "first"),
            venue=("venue", "first"),
            season=("season_year", "first"),
            runs=("runs_total", "sum"),
            wickets=("wicket_fell", "sum"),
            balls=("legal_ball", "sum"),
            fours=("four", "sum"),
            sixes=("six", "sum"),
            dots=("dot", "sum"),
        )
    )
    innings = innings[innings["balls"] >= 60].copy()
    innings["run_rate"] = innings["runs"] * 6 / innings["balls"]
    innings["dot_pct"] = innings["dots"] * 100 / innings["balls"]
    features = ["runs", "wickets", "run_rate", "fours", "sixes", "dot_pct"]
    detector = make_pipeline(StandardScaler(), OneClassSVM(nu=0.02, gamma="scale")).fit(innings[features])
    innings["anomaly_score"] = detector.decision_function(innings[features])
    freak = innings.nsmallest(10, "anomaly_score")
    return [
        {
            "batting_team": row.batting_team,
            "fielding_team": row.fielding_team,
            "venue": row.venue,
            "season": int(row.season),
            "innings": int(row.innings),
            "score": f"{int(row.runs)}/{int(row.wickets)}",
            "run_rate": round(float(row.run_rate), 2),
            "fours": int(row.fours),
            "sixes": int(row.sixes),
            "anomaly_score": round(float(row.anomaly_score), 3),
        }
        for row in freak.itertuples(index=False)
    ]


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    remaining = centroids.copy()
    for label, column, highest in [("Power-hitter", "strike_rate", True), ("Anchor", "average", True), ("Grinder", "dot_pct", True)]:
        cluster = remaining[column].idxmax() if highest else remaining[column].idxmin()
        names[int(cluster)] = label
        remaining = remaining.drop(index=cluster)
    for cluster in remaining.index:
        names[int(cluster)] = "Accumulator"
    return names


def cluster_batters(deliveries: pd.DataFrame) -> dict:
    faced = deliveries[deliveries["extras_wides"] == 0]
    frame = faced.assign(
        four=(faced["runs_batter"] == 4).astype(int),
        six=(faced["runs_batter"] == 6).astype(int),
        dot=(faced["runs_batter"] == 0).astype(int),
        out=(faced["wicket_player_out"].fillna("") == faced["batter"]).astype(int),
    )
    batters = (
        frame.groupby("batter", as_index=False)
        .agg(runs=("runs_batter", "sum"), balls=("batter", "size"), fours=("four", "sum"), sixes=("six", "sum"), dots=("dot", "sum"), outs=("out", "sum"))
    )
    batters = batters[batters["balls"] >= 300].reset_index(drop=True)
    batters["strike_rate"] = batters["runs"] * 100 / batters["balls"]
    batters["average"] = batters["runs"] / batters["outs"].clip(lower=1)
    batters["boundary_pct"] = (batters["fours"] + batters["sixes"]) * 100 / batters["balls"]
    batters["dot_pct"] = batters["dots"] * 100 / batters["balls"]
    features = ["strike_rate", "average", "boundary_pct", "dot_pct"]
    scaled = StandardScaler().fit_transform(batters[features])

    elbow = [
        {"k": k, "inertia": round(float(KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled).inertia_), 2)}
        for k in range(1, 9)
    ]
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(scaled)
    batters["cluster"] = kmeans.labels_
    centroids = batters.groupby("cluster")[features].mean()
    names = name_clusters(centroids)
    batters["style"] = batters["cluster"].map(names)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)
    batters["x"], batters["y"] = coordinates[:, 0], coordinates[:, 1]

    clusters = []
    for cluster, centroid in centroids.iterrows():
        members = batters[batters["cluster"] == cluster].sort_values("runs", ascending=False)
        clusters.append(
            {
                "cluster": int(cluster),
                "style": names[int(cluster)],
                "size": int(len(members)),
                "centroid": {name: round(float(value), 2) for name, value in centroid.items()},
                "top_players": members["batter"].head(5).tolist(),
            }
        )
    return {
        "k": 4,
        "features": features,
        "elbow": elbow,
        "explained_variance": [round(float(value) * 100, 1) for value in pca.explained_variance_ratio_],
        "clusters": clusters,
        "points": [
            {
                "player": row.batter,
                "style": row.style,
                "x": round(float(row.x), 3),
                "y": round(float(row.y), 3),
                "strike_rate": round(float(row.strike_rate), 1),
                "average": round(float(row.average), 1),
            }
            for row in batters.itertuples(index=False)
        ],
    }


def bucket(value: int, edges: list[int]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def coach_phase(balls_left: int) -> int:
    # Overs 1-6 leave 85-120 balls, overs 7-15 leave 31-84, the death leaves 1-30.
    return 0 if balls_left > 84 else 1 if balls_left > 30 else 2


def coach_state(balls_left: int, runs_needed: int, wickets: int) -> tuple[int, int, int]:
    required_rate = runs_needed * 6 / balls_left
    return bucket(balls_left, COACH_BALLS_EDGES), bucket(required_rate, COACH_RATE_EDGES), bucket(wickets, COACH_WICKETS_EDGES)


def outcome_distributions(deliveries: pd.DataFrame) -> np.ndarray:
    """Ball-outcome odds per phase and action, shape (phase, action, outcome)."""
    chase = deliveries[(deliveries["innings"] == 2) & (deliveries["legal_delivery"] == 1) & (deliveries["required_runs_before"] > 0)]
    outcome = np.select(
        [chase["wicket_fell"] == 1, chase["runs_total"] >= 6, chase["runs_total"] >= 4, chase["runs_total"] >= 2, chase["runs_total"] == 1],
        ["W", "6", "4", "2", "1"],
        default="0",
    )
    base = (
        pd.crosstab(chase["phase"].to_numpy(), outcome, normalize="index")
        .reindex(index=COACH_PHASES, columns=COACH_OUTCOMES, fill_value=0.0)
        .to_numpy()
    )
    # Rotate is the real chase distribution for that phase; Defend and Attack trade boundaries for wicket risk.
    shifts = np.array(
        [
            [1.3, 1.3, 1.0, 0.5, 0.3, 0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.8, 0.7, 0.8, 1.5, 2.0, 1.8],
        ]
    )
    distributions = base[:, None, :] * shifts[None, :, :]
    return distributions / distributions.sum(axis=2, keepdims=True)


def simulate_ball(rng: np.random.Generator, cumulative: np.ndarray, balls_left: int, runs_needed: int, wickets: int, action: int) -> tuple[int, int, int, float, bool]:
    odds = cumulative[coach_phase(balls_left), action]
    outcome = int(np.searchsorted(odds, rng.random() * odds[-1]))
    balls_left -= 1
    if outcome == 5:
        wickets -= 1
    else:
        runs_needed -= COACH_RUNS_FOR_OUTCOME[outcome]
    if runs_needed <= 0:
        return balls_left, runs_needed, wickets, 1.0, True
    if wickets == 0 or balls_left == 0:
        return balls_left, runs_needed, wickets, -1.0, True
    return balls_left, runs_needed, wickets, 0.0, False


def evaluate_coach(q_table: np.ndarray, cumulative: np.ndarray, targets: list[int]) -> float:
    """Win rate when the coach plays full 120-ball chases greedily, with a fixed random seed for a fair comparison."""
    rng = np.random.default_rng(7)
    wins = 0
    for target in targets:
        balls_left, runs_needed, wickets, done, reward = 120, target, 10, False, 0.0
        while not done:
            action = int(q_table[coach_state(balls_left, runs_needed, wickets)].argmax())
            balls_left, runs_needed, wickets, reward, done = simulate_ball(rng, cumulative, balls_left, runs_needed, wickets, action)
        wins += reward > 0
    return wins / len(targets)


def train_chase_coach(deliveries: pd.DataFrame, episodes: int = 150000) -> dict:
    probabilities = outcome_distributions(deliveries)
    cumulative = probabilities.cumsum(axis=2)
    rng = np.random.default_rng(RANDOM_STATE)
    q_table = np.zeros((len(COACH_BALLS_BUCKETS), len(COACH_RATE_BUCKETS), len(COACH_WICKETS_BUCKETS), len(COACH_ACTIONS)))
    visits = np.zeros_like(q_table)
    test_targets = [int(target) for target in np.random.default_rng(11).integers(140, 221, size=300)]
    learning_curve = [{"episode": 0, "win_rate": round(evaluate_coach(q_table, cumulative, test_targets) * 100, 1)}]

    for episode in range(1, episodes + 1):
        # Curriculum: learn the end of the chase first, then start episodes earlier and earlier.
        latest_start = 30 + int(90 * min(1.0, episode / (0.5 * episodes)))
        balls_left = int(rng.integers(1, latest_start + 1))
        runs_needed = max(1, round(balls_left * float(rng.uniform(4, 16)) / 6))
        if rng.random() < 0.3:
            wickets = int(rng.integers(1, 11))
        else:
            wickets = int(rng.integers(max(1, 10 - (120 - balls_left) // 12 - 2), 11))
        done = False
        while not done:
            state = coach_state(balls_left, runs_needed, wickets)
            # Explore more in situations the coach has rarely seen.
            epsilon = max(0.05, 20 / (20 + visits[state].sum()))
            action = int(rng.integers(3)) if rng.random() < epsilon else int(q_table[state].argmax())
            balls_left, runs_needed, wickets, reward, done = simulate_ball(rng, cumulative, balls_left, runs_needed, wickets, action)
            target = reward if done else q_table[coach_state(balls_left, runs_needed, wickets)].max()
            visits[state + (action,)] += 1
            # A floor on the learning rate lets values forget early guesses as later situations get learned.
            alpha = max(0.05, 1 / visits[state + (action,)])
            q_table[state + (action,)] += alpha * (target - q_table[state + (action,)])
        if episode % 10000 == 0:
            learning_curve.append({"episode": episode, "win_rate": round(evaluate_coach(q_table, cumulative, test_targets) * 100, 1)})

    policy = []
    for b, balls_label in enumerate(COACH_BALLS_BUCKETS):
        for r, rate_label in enumerate(COACH_RATE_BUCKETS):
            for w, wickets_label in enumerate(COACH_WICKETS_BUCKETS):
                values = q_table[b, r, w]
                policy.append(
                    {
                        "balls_left": balls_label,
                        "required_rate": rate_label,
                        "wickets_in_hand": wickets_label,
                        "best_action": COACH_ACTIONS[int(values.argmax())],
                        "q_values": {action: round(float(value), 3) for action, value in zip(COACH_ACTIONS, values)},
                        "visits": int(visits[b, r, w].sum()),
                    }
                )
    return {
        "episodes": episodes,
        "actions": COACH_ACTIONS,
        "balls_buckets": COACH_BALLS_BUCKETS,
        "rate_buckets": COACH_RATE_BUCKETS,
        "wickets_buckets": COACH_WICKETS_BUCKETS,
        "balls_edges": COACH_BALLS_EDGES,
        "rate_edges": COACH_RATE_EDGES,
        "wickets_edges": COACH_WICKETS_EDGES,
        "outcome_probabilities": {
            phase: {
                action: {outcome: round(float(value), 4) for outcome, value in zip(COACH_OUTCOMES, row)}
                for action, row in zip(COACH_ACTIONS, phase_rows)
            }
            for phase, phase_rows in zip(COACH_PHASES, probabilities)
        },
        "policy": policy,
        "test_chases": len(test_targets),
        "learning_curve": learning_curve,
    }


def train_insights() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    deliveries = load_processed("deliveries")
    matches = load_processed("matches")

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "charts": build_eda_charts(deliveries, matches),
        "score_models": train_score_models(deliveries),
        "chase_models": train_chase_models(deliveries),
        "freak_innings": detect_freak_innings(deliveries),
        "batter_styles": cluster_batters(deliveries),
        "chase_coach": train_chase_coach(deliveries),
    }
    (REPORTS_DIR / "model_insights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    train_insights()
