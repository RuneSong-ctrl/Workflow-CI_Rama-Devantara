import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    print("Memuat data steam_ready_to_train.csv...")
    df = pd.read_csv('steam_ready_to_train.csv')
    
    X = df.drop(columns=['Peak CCU'])
    y = df['Peak CCU']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hapus folder model_output jika sudah ada dari run sebelumnya
    if os.path.exists("model_output"):
        shutil.rmtree("model_output")

    with mlflow.start_run(run_name="CI_Run"):
        print("Melatih model Random Forest...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan model secara lokal untuk di-build oleh Docker 
        mlflow.sklearn.save_model(model, "model_output")
        print("Model berhasil disimpan di folder 'model_output'.")