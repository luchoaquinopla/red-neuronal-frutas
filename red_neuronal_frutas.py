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

def procesar_frutas_nuevas(model, scaler):
    """Carga, procesa y hace predicciones sobre frutas nuevas sin etiquetas"""
    print("\n" + "="*50)
    print("PROCESANDO FRUTAS NUEVAS")
    print("="*50)
    
    try:
        # Cargar frutas nuevas
        print("Cargando frutas nuevas...")
        frutas_nuevas = pd.read_csv('frutas_nuevas.csv')
        print(f"Frutas nuevas cargadas: {frutas_nuevas.shape[0]} muestras")
        
        # Verificar columnas
        if not all(col in frutas_nuevas.columns for col in ['color', 'firmness']):
            print("❌ Error: El archivo debe contener columnas 'color' y 'firmness'")
            return None
        
        # Extraer características
        X_nuevas = frutas_nuevas[['color', 'firmness']].values
        print(f"Características extraídas: {X_nuevas.shape}")
        
        # Escalar usando el mismo scaler entrenado
        print("Escalando características con el scaler entrenado...")
        X_nuevas_scaled = scaler.transform(X_nuevas)
        
        # Hacer predicciones
        print("Realizando predicciones...")
        predicciones_proba = model.predict(X_nuevas_scaled, verbose=0)
        predicciones = (predicciones_proba > 0.5).astype(int).flatten()
        
        # Agregar predicciones al DataFrame
        frutas_nuevas['prediccion'] = predicciones
        frutas_nuevas['probabilidad'] = predicciones_proba.flatten()
        
        # Mostrar resultados
        print("\nPrimeras 10 frutas nuevas con predicciones:")
        print(frutas_nuevas.head(10))
        
        # Estadísticas de predicciones
        conteo_predicciones = frutas_nuevas['prediccion'].value_counts()
        print(f"\nResumen de predicciones:")
        print(f"Frutas predichas como No Maduras (0): {conteo_predicciones.get(0, 0)}")
        print(f"Frutas predichas como Maduras (1): {conteo_predicciones.get(1, 0)}")
        
        return frutas_nuevas, X_nuevas, predicciones
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'frutas_nuevas.csv'")
        print("💡 Creando archivo de ejemplo...")
        return None
    except Exception as e:
        print(f"❌ Error al procesar frutas nuevas: {e}")
        return None

def grafico_frutas_nuevas(frutas_nuevas, X_nuevas, predicciones):
    """Genera gráfico de dispersión de frutas nuevas con predicciones"""
    plt.figure(figsize=(12, 8))
    
    # Definir colores para las predicciones
    colores = ['#ff4444', '#44ff44']  # Rojo para No Madura, Verde para Madura
    etiquetas = ['No Madura', 'Madura']
    
    # Crear el gráfico de dispersión
    scatter = plt.scatter(X_nuevas[:, 0], X_nuevas[:, 1], 
                         c=predicciones, 
                         cmap='RdYlGn',
                         edgecolors='black', 
                         s=100, 
                         alpha=0.8)
    
    plt.xlabel('Color')
    plt.ylabel('Firmeza')
    plt.title('Predicciones de Madurez - Frutas Nuevas')
    
    # Crear leyenda personalizada
    handles = [plt.scatter([], [], c=colores[i], s=100, edgecolors='black', 
                          alpha=0.8, label=etiquetas[i]) 
               for i in range(len(etiquetas))]
    plt.legend(handles=handles, title='Predicción', loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ Gráfico de frutas nuevas generado")

def validar_predicciones_frutas_nuevas(model, scaler):
    """Valida las predicciones usando un dataset con etiquetas reales"""
    print("\n" + "="*50)
    print("VALIDACIÓN DE PREDICCIONES")
    print("="*50)
    
    try:
        # Cargar dataset de validación con etiquetas reales
        print("Cargando dataset de validación con etiquetas reales...")
        frutas_validacion = pd.read_csv('frutas_validacion.csv')
        print(f"Dataset de validación cargado: {frutas_validacion.shape[0]} muestras")
        
        # Separar características y etiquetas reales
        X_val = frutas_validacion[['color', 'firmness']].values
        y_real = frutas_validacion['label'].values
        
        # Escalar características
        X_val_scaled = scaler.transform(X_val)
        
        # Hacer predicciones
        predicciones_proba = model.predict(X_val_scaled, verbose=0)
        predicciones = (predicciones_proba > 0.5).astype(int).flatten()
        
        # Calcular métricas de validación
        accuracy = accuracy_score(y_real, predicciones)
        cm = confusion_matrix(y_real, predicciones)
        
        print(f"\n📊 RESULTADOS DE VALIDACIÓN:")
        print(f"Accuracy en frutas nuevas: {accuracy:.4f}")
        print(f"Matriz de confusión:")
        print(cm)
        
        # Análisis detallado
        print(f"\n📋 ANÁLISIS DETALLADO:")
        
        # Comparar predicciones vs reales
        df_comparacion = pd.DataFrame({
            'color': frutas_validacion['color'],
            'firmness': frutas_validacion['firmness'],
            'real': y_real,
            'prediccion': predicciones,
            'probabilidad': predicciones_proba.flatten(),
            'correcto': y_real == predicciones
        })
        
        print("\nPrimeras 10 comparaciones:")
        print(df_comparacion.head(10))
        
        # Estadísticas de errores
        errores = df_comparacion[~df_comparacion['correcto']]
        if len(errores) > 0:
            print(f"\n❌ ERRORES DE PREDICCIÓN ({len(errores)} errores):")
            print(errores)
        else:
            print(f"\n✅ PERFECTO: Todas las predicciones fueron correctas!")
        
        # Análisis de confianza
        print(f"\n🎯 ANÁLISIS DE CONFIANZA:")
        prob_baja_confianza = df_comparacion[
            (df_comparacion['probabilidad'] > 0.3) & 
            (df_comparacion['probabilidad'] < 0.7)
        ]
        
        if len(prob_baja_confianza) > 0:
            print(f"Predicciones con baja confianza (probabilidad entre 0.3-0.7): {len(prob_baja_confianza)}")
            print(prob_baja_confianza)
        else:
            print("✅ Todas las predicciones tienen alta confianza")
        
        return df_comparacion, accuracy
        
    except FileNotFoundError:
        print("❌ Error: No se encontró 'frutas_validacion.csv'")
        print("💡 Este archivo contiene las etiquetas reales para validar las predicciones")
        return None, None
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return None, None

def analizar_casos_extremos(model, scaler):
    """Analiza casos extremos para validar el comportamiento del modelo"""
    print("\n" + "="*50)
    print("ANÁLISIS DE CASOS EXTREMOS")
    print("="*50)
    
    # Crear casos extremos conocidos
    casos_extremos = {
        'Muy No Madura': [1.0, 1.0],      # Valores muy bajos
        'Muy Madura': [8.0, 8.0],         # Valores muy altos
        'Color Alto, Firmeza Baja': [7.0, 2.0],  # Caso contradictorio
        'Color Bajo, Firmeza Alta': [2.0, 7.0],  # Caso contradictorio
        'Valores Intermedios': [4.0, 4.0] # Punto medio
    }
    
    print("Probando casos extremos conocidos:")
    
    for nombre, valores in casos_extremos.items():
        # Escalar el caso
        valores_escalados = scaler.transform([valores])
        
        # Predecir
        prob = model.predict(valores_escalados, verbose=0)[0][0]
        prediccion = 1 if prob > 0.5 else 0
        
        print(f"  {nombre}: [{valores[0]:.1f}, {valores[1]:.1f}] → "
              f"Predicción: {'Madura' if prediccion else 'No Madura'} "
              f"(prob: {prob:.3f})")
    
    print("\n✅ Análisis de casos extremos completado")

def grafico_validacion_con_frontera(X_val, y_real, predicciones, model, scaler):
    """Genera gráfico que muestra predicciones vs reales con frontera de decisión"""
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Etiquetas reales
    plt.subplot(1, 2, 1)
    
    # Crear malla para frontera de decisión
    x_min, x_max = X_val[:, 0].min() - 0.5, X_val[:, 0].max() + 0.5
    y_min, y_max = X_val[:, 1].min() - 0.5, X_val[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predecir en la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Frontera de decisión
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probabilidad de Madurez')
    
    # Puntos reales
    scatter1 = plt.scatter(X_val[:, 0], X_val[:, 1], c=y_real, cmap='RdYlBu', 
                          edgecolors='black', s=100, alpha=0.8)
    plt.xlabel('Color')
    plt.ylabel('Firmeza')
    plt.title('Etiquetas Reales vs Frontera de Decisión')
    plt.legend(handles=scatter1.legend_elements()[0], 
              labels=['No Madura', 'Madura'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Predicciones
    plt.subplot(1, 2, 2)
    
    # Misma frontera
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probabilidad de Madurez')
    
    # Puntos predichos (coloreados según predicción)
    scatter2 = plt.scatter(X_val[:, 0], X_val[:, 1], c=predicciones, cmap='RdYlBu', 
                          edgecolors='black', s=100, alpha=0.8)
    plt.xlabel('Color')
    plt.ylabel('Firmeza')
    plt.title('Predicciones vs Frontera de Decisión')
    plt.legend(handles=scatter2.legend_elements()[0], 
              labels=['No Madura', 'Madura'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Gráfico de validación con frontera generado")

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
    
    # 7. Procesar frutas nuevas
    resultado_frutas_nuevas = procesar_frutas_nuevas(model, scaler)
    if resultado_frutas_nuevas is not None:
        frutas_nuevas, X_nuevas, predicciones = resultado_frutas_nuevas
        grafico_frutas_nuevas(frutas_nuevas, X_nuevas, predicciones)
    
    # 8. Validar predicciones con dataset de prueba
    df_validacion, accuracy_validacion = validar_predicciones_frutas_nuevas(model, scaler)
    
    # 9. Análisis de casos extremos
    analizar_casos_extremos(model, scaler)
    
    # 10. Gráfico de validación si hay datos de validación
    if df_validacion is not None:
        X_val = df_validacion[['color', 'firmness']].values
        y_real = df_validacion['real'].values
        predicciones_val = df_validacion['prediccion'].values
        grafico_validacion_con_frontera(X_val, y_real, predicciones_val, model, scaler)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print(f"Accuracy final: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
