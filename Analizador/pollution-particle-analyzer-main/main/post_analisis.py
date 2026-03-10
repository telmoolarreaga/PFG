from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN

def scientific_pipeline(features_df):
    # 1️⃣ Escalar features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features_df)

    # 2️⃣ Eliminar features con varianza muy baja
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X_scaled)

    # 3️⃣ Reducir dimensionalidad con PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_selected)

    # 4️⃣ Clustering de partículas
    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_pca)

    return X_pca, labels