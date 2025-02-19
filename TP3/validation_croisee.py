from scipy.linalg import svd
import numpy as np 

def train_svd(R, n_dimensions):
    '''
      ## L'entraînement du modèle de prédiction consiste à factoriser avec SVD la matrice de votes R
    '''
    u, s, vt = svd(R)
    return u[:, :n_dimensions], s[:n_dimensions], vt[:n_dimensions, :]

def validation_croisee(R, R_imputation, test_indices, valeurs_observees, n_dimensions):
    '''
      ## Validation croisée
      ## R : matrice de votes originale
      ## R_imputation : valeurs d'imputation (mêmes dimensions que R)
      ## test_indices : index des cellules test de R à retirer de l'entraînement
      ## valeurs_observees : matrice booleenne des cellules observees de R
      ## n_dimensions : nombre de dimensions latentes pour la factorisation
    '''

    R_entrainement = R_imputation.copy()
    R_entrainement[valeurs_observees] = R[valeurs_observees]
    R_entrainement.ravel()[test_indices] = R_imputation.ravel()[test_indices]
    R_svd = train_svd(R_entrainement, n_dimensions)
    Rhat = predict_svd(R_svd, n_dimensions)
    R_test = R.ravel()[test_indices]
    Rhat_test = Rhat.ravel()[test_indices]

    rmse = np.sqrt(np.nanmean((Rhat_test - R_test)**2))

    return rmse

def validation_croisee_replis(R, R_imputation, echantillon_replis, valeurs_observees, n_dimensions):
    '''
      ## Routine pour créer une série de replis
    '''
    return np.apply_along_axis(lambda test: validation_croisee(R, R_imputation, test, valeurs_observees, n_dimensions), axis=1, arr=echantillon_replis)

def predict_svd(R_svd, n_dimensions):
    '''
      ## Prédiction de votes à partir d'une factorisation avec nDimensions
    '''
    u, s, vt = R_svd
    return u @ np.diag(s) @ vt
