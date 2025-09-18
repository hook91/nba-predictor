import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Fonction pour récupérer les stats d’un joueur
# -------------------------
def get_player_stats(player_name, season="2023-24"):
    player = [p for p in players.get_players() if p["full_name"].lower() == player_name.lower()]
    if not player:
        return None
    player_id = player[0]["id"]
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    return df

# -------------------------
# Fonction pour entraîner un modèle simple
# -------------------------
def train_predictor(data, stat="PTS"):
    if data is None or data.empty:
        return None, None
    df = data[[stat]].copy()
    df["lag1"] = df[stat].shift(1)
    df["lag3"] = df[stat].rolling(3).mean().shift(1)
    df["lag5"] = df[stat].rolling(5).mean().shift(1)
    df = df.dropna()
    
    if df.empty:
        return None, None
    
    X = df[["lag1", "lag3", "lag5"]]
    y = df[stat]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.tail(1)

# -------------------------
# Fonction de prédiction
# -------------------------
def predict_next(model, last_features):
    return model.predict(last_features)[0]

# -------------------------
# Interface Streamlit
# -------------------------
st.set_page_config(page_title="NBA Predictor", layout="centered")
st.title("🏀 NBA Predictor (Stats Joueur)")

player_name = st.text_input("Nom du joueur (ex: LeBron James)", "LeBron James")
season = st.text_input("Saison NBA (format 2023-24)", "2023-24")

if st.button("Prédire stats joueur"):
    data = get_player_stats(player_name, season)
    if data is None:
        st.error("Joueur introuvable ❌")
    else:
        st.write("📊 Derniers matchs :", data.head())

        results = {}
        for stat in ["PTS", "REB", "FG3M"]:
            model, last_feats = train_predictor(data, stat)
            if model:
                prediction = predict_next(model, last_feats)
                results[stat] = round(prediction, 1)
        
        st.subheader(f"📌 Prédictions pour {player_name}")
        for k, v in results.items():
            st.write(f"**{k} prévu :** {v}")

st.markdown("---")
st.info("Cette app utilise `nba_api` et un modèle RandomForest pour prédire les stats de joueur.")
