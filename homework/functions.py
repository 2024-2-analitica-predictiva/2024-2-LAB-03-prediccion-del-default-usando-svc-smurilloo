def load_data(file_path):
    import pandas as pd
    import os

    # Verifica si los archivos existen antes de leerlos
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo de prueba no se encuentra: {file_path}")
    
    # Cargar los archivos CSV comprimidos
    data= pd.read_csv(file_path, index_col=False, compression="zip")
 
    return data

def clean_data(data):
    import numpy as np
    df = data.copy()  # Crear una copia del dataframe original

    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns='ID', inplace=True)
 
    df = df.iloc[df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)].index]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    return df  # Devolver el dataframe limpio¡


def make_pipeline(estimator):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif

    categorical_feature=['EDUCATION','SEX','MARRIAGE']
    
   
  
    # Crear el transformador para las columnas categóricas y numericas 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature)
            
        ],
        remainder='passthrough' 
  
    )
    # Descomposición PCA
    pca_transformer = PCA()
    ## selecciona las mejores k variables 
    k_best_selector = SelectKBest(score_func=f_classif, k=1)


    # Crear el pipeline con preprocesamiento y el modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', pca_transformer),
        ('num', StandardScaler()),
        ('kbest', k_best_selector),
        ('estimator', estimator)  # Establecer el estimador que se pasa como argumento
    ],
    verbose=False)

    return pipeline


def save_estimator_compressed(estimator, file_path="../files/models/model.pkl.gz"):
    import os
    import gzip
    import pickle
    # Asegurarse de que el directorio exista
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Guardar el modelo comprimido
    with gzip.open(file_path, "wb") as file:
        pickle.dump(estimator, file)



def load_estimator_compressed(file_path="../files/models/model.pkl.gz"):
    import os
    import gzip
    import pickle
    try:
        # Asegurarse de que el directorio exista
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Verificar si el archivo existe antes de intentar abrirlo
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no se encuentra.")
        
        # Abrir el archivo comprimido en modo de lectura binaria
        with gzip.open(file_path, "rb") as file:
            estimator = pickle.load(file)
        
        return estimator

    except Exception as e:
        print(f"Ocurrió un error al cargar el modelo: {e}")
        return None
    
def make_grid_search(estimator, param_grid, cv=10):

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2

    )

    return grid_search 

def calculate_and_save_metrics(model, x_train, x_test, y_train, y_test):
    from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    
    )
    import json
    import os
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        }
    ]

    os.makedirs("../files/output", exist_ok=True)
    with open("../files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

def calculate_and_save_confusion_matrices(model, x_train, x_test, y_train, y_test):
    import json
    from sklearn.metrics import confusion_matrix

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    matrices = [
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    with open("../files/output/metrics.json", "a") as f:
        for matrix in matrices:
            f.write(json.dumps(matrix) + '\n')

def print_metric(y_true, y_pred, dataset):
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Métricas para el dataset {dataset}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('-' * 40)


# Paso 4: Imprimir la matriz de confusión para ambos conjuntos
def print_confusion_matrix(y_true, y_pred, dataset):
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_true, y_pred)
    print(f'Matriz de Confusión para el dataset {dataset}:')
    print(matriz)
    print('-' * 40)