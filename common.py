import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 15
NUMERIC = ['Age', 'Height', 'Weight', 'BMI', 'PriorMedals']
CATEGORICAL = ['Event_encoded', 'Sex_encoded']
FEATURES = NUMERIC + CATEGORICAL


def load_data(path='athlete_events.csv'):
    df = pd.read_csv(path)
    print("Initial dataset size:", df.shape)

    # create binary label
    df['MedalBinary'] = df['Medal'].notnull().astype(int)

    # medals the athlete won at earlier Games (strictly earlier years only, so
    # nothing from the Games being predicted). Computed before dropping rows so
    # medals from entries with missing height/weight still count.
    medals_per_games = df.groupby(['ID', 'Year'])['MedalBinary'].sum()
    prior_medals = medals_per_games.groupby(level='ID').cumsum() - medals_per_games
    df = df.merge(prior_medals.rename('PriorMedals').reset_index(), on=['ID', 'Year'], how='left')

    # drop rows with missing values
    df = df.dropna(subset=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event'])
    print("After dropping missing values:", df.shape)

    # create features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
    df['Event_encoded'] = LabelEncoder().fit_transform(df['Event'])
    return df


def split_train_test(df):
    # all 2016 athletes, including returning ones: their earlier results are
    # known before the Games, so using them (via PriorMedals) isn't leakage
    train_df = df[df['Year'] < 2016]
    test_df = df[df['Year'] == 2016]
    return train_df, test_df


def make_sampler(method):
    if method == 'SMOTE':
        # SMOTENC picks categorical values from neighbours instead of interpolating
        # label-encoded IDs, which would create e.g. women in men's events
        return SMOTENC(categorical_features=CATEGORICAL, random_state=RANDOM_STATE)
    return RandomUnderSampler(random_state=RANDOM_STATE)


def make_pipeline(model, method, params=None, n_jobs=-1):
    """Resampler + classifier. Resampling sits inside the pipeline so it only
    ever touches the data a model is fitted on, never validation/test folds."""
    params = params or {}
    steps = [('resample', make_sampler(method))]

    if model == 'LogisticRegression':
        # event IDs have no order, so one-hot them for the linear model
        steps.append(('prep', ColumnTransformer([
            ('num', StandardScaler(), NUMERIC),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ])))
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
    elif model == 'RandomForest':
        clf = RandomForestClassifier(n_jobs=n_jobs, random_state=RANDOM_STATE, **params)
    elif model == 'XGBoost':
        clf = XGBClassifier(eval_metric='logloss', n_jobs=n_jobs, random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"Unknown model: {model}")

    steps.append(('clf', clf))
    return Pipeline(steps)
