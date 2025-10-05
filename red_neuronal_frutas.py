import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos en español
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

def cargar_y_preprocesar_datos():
    """Carga y preprocesa el dataset de frutas"""
    print("Cargando dataset de frutas...")
    
    # Cargar datos
    data = pd.read_csv('frutas_sinteticas_1000.csv')
    print(f"Dataset cargado: {data.shape[0]} muestras, {data.shape[1]} características")
    
    # Separar características y etiquetas
    X = data[['color', 'firmness']].values
    y = data['label'].values
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, data

def dividir_datos(X, y):
    """Divide los datos en entrenamiento (75%) y prueba (25%)"""
    print("Dividiendo datos en entrenamiento (75%) y prueba (25%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def crear_modelo_mlp():
    """Crea la red neuronal multicapa"""
    print("Creando red neuronal multicapa...")
    
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(2,), name='capa_oculta'),
        layers.Dense(1, activation='sigmoid', name='capa_salida')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Arquitectura del modelo:")
    model.summary()
    
    return model

def entrenar_modelo(model, X_train, y_train, X_test, y_test):
    """Entrena el modelo y muestra el progreso"""
    print("Entrenando modelo...")
    
    # Configurar callback para early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluar_modelo(model, X_test, y_test):
    """Evalúa el modelo y muestra métricas"""
    print("\nEvaluando modelo...")
    
    # Predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(cm)
    
    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Madura', 'Madura']))
    
    return y_pred, accuracy, cm

def grafico_dispersion_frontera(X, y, model, scaler, titulo="Datos de Frutas con Frontera de Decisión"):
    """Crea gráfico de dispersión con frontera de decisión"""
    plt.figure(figsize=(12, 8))
    
    # Crear malla para frontera de decisión
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predecir en la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Gráfico de contorno para frontera de decisión
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probabilidad de Madurez')
    
    # Puntos de datos
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=50, alpha=0.8)
    
    plt.xlabel('Color (escalado)')
    plt.ylabel('Firmeza (escalado)')
    plt.title(titulo)
    plt.legend(handles=scatter.legend_elements()[0], 
              labels=['No Madura', 'Madura'], loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def grafico_distribucion_clases(data):
    """Crea gráfico de barras de distribución de clases"""
    plt.figure(figsize=(8, 6))
    
    # Contar clases
    conteo = data['label'].value_counts()
    etiquetas = ['No Madura', 'Madura']
    
    # Gráfico de barras
    barras = plt.bar(etiquetas, conteo.values, color=['#ff7f7f', '#7fbf7f'], 
                    edgecolor='black', alpha=0.7)
    
    # Agregar valores en las barras
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura + 0.5,
                f'{int(altura)}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Cantidad de Frutas')
    plt.title('Distribución de Frutas Maduras y No Maduras')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def grafico_historial_entrenamiento(history):
    """Muestra el historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Accuracy del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal que ejecuta todo el pipeline"""
    print("="*60)
    print("RED NEURONAL PARA CLASIFICACIÓN DE FRUTAS MADURAS")
    print("="*60)
    
    # 1. Cargar y preprocesar datos
    X, y, scaler, data = cargar_y_preprocesar_datos()
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # 3. Crear modelo
    model = crear_modelo_mlp()
    
    # 4. Entrenar modelo
    history = entrenar_modelo(model, X_train, y_train, X_test, y_test)
    
    # 5. Evaluar modelo
    y_pred, accuracy, cm = evaluar_modelo(model, X_test, y_test)
    
    # 6. Gráficos
    print("\nGenerando gráficos...")
    
    # Gráfico de distribución de clases
    grafico_distribucion_clases(data)
    
    # Gráfico de dispersión con frontera de decisión (datos originales)
    X_original = data[['color', 'firmness']].values
    grafico_dispersion_frontera(X_original, y, model, scaler, 
                               "Frontera de Decisión - Datos Originales")
    
    # Gráfico de dispersión con frontera de decisión (datos escalados)
    grafico_dispersion_frontera(X, y, model, StandardScaler().fit(X), 
                               "Frontera de Decisión - Datos Escalados")
    
    # Historial de entrenamiento
    grafico_historial_entrenamiento(history)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print(f"Accuracy final: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
