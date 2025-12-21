import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load dataset mentah"""
    return pd.read_csv(path)

def preprocessing(df):
    """Preprocessing data sesuai eksperimen notebook"""

    # =====================
    # 1. Encoding target RiskLevel
    # =====================
    risk_mapping = {
        'low risk': 0,
        'mid risk': 1,
        'high risk': 2
    }

    df['RiskLevel_encoded'] = df['RiskLevel'].map(risk_mapping)

    # =====================
    # 2. Drop kolom target asli
    # =====================
    df = df.drop(columns=['RiskLevel'])

    # =====================
    # 3. Split fitur dan target
    # =====================
    X = df.drop(columns=['RiskLevel_encoded'])
    y = df['RiskLevel_encoded']

    # =====================
    # 4. Scaling fitur numerik
    # =====================
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # =====================
    # 5. Gabungkan kembali
    # =====================
    final_df = pd.concat([X, y], axis=1)

    return final_df

def save_data(df, path):
    """Simpan dataset hasil preprocessing"""
    df.to_csv(path, index=False)

if __name__ == "__main__":
    raw_path = "maternal_health_risk_dataset_raw.csv"
    output_path = "preprocessing/dataset_preprocessed.csv"

    df_raw = load_data(raw_path)
    df_clean = preprocessing(df_raw)
    save_data(df_clean, output_path)

    print("Preprocessing otomatis selesai. Dataset siap dilatih.")
