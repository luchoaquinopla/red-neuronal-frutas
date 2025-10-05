# Documentación de Funciones - Red Neuronal para Clasificación de Frutas

## Importaciones y Configuración Inicial

```python
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
```

**Propósito**: Importa todas las librerías necesarias para el proyecto:

- **pandas**: Manipulación de datos CSV
- **numpy**: Operaciones matemáticas y arrays
- **matplotlib/seaborn**: Creación de gráficos
- **sklearn**: Herramientas de machine learning (división de datos, escalado, métricas)
- **tensorflow/keras**: Construcción y entrenamiento de redes neuronales
- **warnings**: Suprime mensajes de advertencia para una salida más limpia

---

## 1. Función `cargar_y_preprocesar_datos()`

```python
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
```

**¿Para qué sirve?**

- Carga el dataset CSV con 1000 muestras de frutas
- Separa las características (color, firmness) de las etiquetas (label)
- Normaliza los datos usando StandardScaler para que tengan media 0 y desviación estándar 1

**¿Cómo funciona?**

1. Lee el archivo CSV usando pandas
2. Extrae las columnas 'color' y 'firmness' como características (X)
3. Extrae la columna 'label' como etiquetas (y)
4. Crea un StandardScaler y ajusta los datos para normalizarlos
5. Retorna los datos escalados, etiquetas, el scaler y los datos originales

**¿Por qué es importante el escalado?**

- Las redes neuronales funcionan mejor cuando los datos están en rangos similares
- Evita que una característica domine sobre otra por tener valores más grandes

---

## 2. Función `dividir_datos(X, y)`

```python
def dividir_datos(X, y):
    """Divide los datos en entrenamiento (75%) y prueba (25%)"""
    print("Dividiendo datos en entrenamiento (75%) y prueba (25%)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

    return X_train, X_test, y_train, y_test
```

**¿Para qué sirve?**

- Divide los datos en dos conjuntos: entrenamiento (75%) y prueba (25%)
- Garantiza que la proporción de clases se mantenga en ambos conjuntos

**¿Cómo funciona?**

1. Usa `train_test_split` de sklearn para dividir aleatoriamente
2. `test_size=0.25`: 25% para prueba, 75% para entrenamiento
3. `random_state=42`: Fija la semilla aleatoria para resultados reproducibles
4. `stratify=y`: Mantiene la proporción de clases (maduras/no maduras) en ambos conjuntos

**¿Por qué dividir los datos?**

- **Entrenamiento**: Para que el modelo aprenda los patrones
- **Prueba**: Para evaluar qué tan bien generaliza el modelo a datos nuevos
- **Estratificación**: Evita sesgos en la evaluación

---

## 3. Función `crear_modelo_mlp()`

```python
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
```

**¿Para qué sirve?**

- Construye la arquitectura de la red neuronal multicapa (MLP)
- Configura el optimizador, función de pérdida y métricas

**¿Cómo funciona?**

1. **Capa oculta**: 10 neuronas con activación ReLU
   - ReLU: f(x) = max(0, x) - solo activa neuronas con valores positivos
   - Input shape: (2,) porque tenemos 2 características (color, firmness)
2. **Capa de salida**: 1 neurona con activación sigmoide
   - Sigmoide: f(x) = 1/(1 + e^(-x)) - produce probabilidades entre 0 y 1
3. **Compilación**:
   - **Optimizador Adam**: Algoritmo de optimización adaptativo
   - **Binary crossentropy**: Función de pérdida para clasificación binaria
   - **Accuracy**: Métrica para evaluar el rendimiento

**¿Por qué esta arquitectura?**

- **ReLU**: Evita el problema de desvanecimiento del gradiente
- **Sigmoide**: Perfecta para clasificación binaria (madura/no madura)
- **Adam**: Optimizador robusto que se adapta a diferentes tipos de datos

---

## 4. Función `entrenar_modelo(model, X_train, y_train, X_test, y_test)`

```python
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
```

**¿Para qué sirve?**

- Entrena la red neuronal usando los datos de entrenamiento
- Implementa early stopping para evitar sobreajuste
- Monitorea el progreso con datos de validación

**¿Cómo funciona?**

1. **Early Stopping**:
   - Monitorea `val_loss` (pérdida en validación)
   - Si no mejora por 20 épocas, detiene el entrenamiento
   - Restaura los mejores pesos encontrados
2. **Entrenamiento**:
   - `epochs=100`: Máximo 100 iteraciones completas
   - `batch_size=8`: Procesa 8 muestras a la vez
   - `validation_data`: Usa datos de prueba para validación
   - `verbose=1`: Muestra el progreso

**¿Por qué early stopping?**

- Evita el sobreajuste (memorizar datos de entrenamiento)
- Encuentra el punto óptimo de entrenamiento
- Mejora la generalización del modelo

---

## 5. Función `evaluar_modelo(model, X_test, y_test)`

```python
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
```

**¿Para qué sirve?**

- Evalúa el rendimiento del modelo entrenado
- Calcula métricas de clasificación
- Genera reportes detallados

**¿Cómo funciona?**

1. **Predicciones**:
   - Obtiene probabilidades del modelo
   - Convierte a clases usando umbral 0.5 (>0.5 = madura)
2. **Accuracy**: Porcentaje de predicciones correctas
3. **Matriz de confusión**:
   - Verdadero Negativo | Falso Positivo
   - Falso Negativo | Verdadero Positivo
4. **Reporte de clasificación**: Precision, Recall, F1-score para cada clase

**¿Qué significan las métricas?**

- **Accuracy**: Precisión general del modelo
- **Precision**: De las frutas predichas como maduras, ¿cuántas lo son realmente?
- **Recall**: De las frutas realmente maduras, ¿cuántas detectó el modelo?
- **F1-score**: Media armónica entre precision y recall

---

## 6. Función `grafico_dispersion_frontera(X, y, model, scaler, titulo)`

```python
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
```

**¿Para qué sirve?**

- Visualiza los datos de frutas en un diagrama de dispersión
- Muestra la frontera de decisión aprendida por la red neuronal
- Permite ver cómo el modelo clasifica diferentes regiones del espacio de características

**¿Cómo funciona?**

1. **Crear malla**: Genera una cuadrícula de puntos en el espacio de características
2. **Predicciones**: Para cada punto de la malla, predice la probabilidad de madurez
3. **Contorno de decisión**: Dibuja regiones coloreadas según la probabilidad
4. **Puntos de datos**: Superpone los datos reales con colores según su clase real
5. **Leyenda**: Indica qué colores representan cada clase

**¿Qué nos dice la frontera de decisión?**

- **Regiones rojas**: Alta probabilidad de frutas no maduras
- **Regiones azules**: Alta probabilidad de frutas maduras
- **Transición**: Área donde el modelo tiene incertidumbre
- **Separación**: Qué tan bien separa el modelo las dos clases

---

## 7. Función `grafico_distribucion_clases(data)`

```python
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
```

**¿Para qué sirve?**

- Muestra la distribución de clases en el dataset
- Permite verificar si hay balance entre clases
- Facilita la interpretación del dataset

**¿Cómo funciona?**

1. Cuenta cuántas frutas hay de cada clase
2. Crea barras con colores diferentes para cada clase
3. Agrega valores numéricos encima de cada barra
4. Aplica formato con leyendas y grid

**¿Por qué es importante?**

- **Balance de clases**: Si hay muchas más frutas de una clase, el modelo puede sesgarse
- **Interpretación**: Ayuda a entender la naturaleza del problema
- **Validación**: Confirma que los datos están bien cargados

---

## 8. Función `grafico_historial_entrenamiento(history)`

```python
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
```

**¿Para qué sirve?**

- Visualiza cómo evolucionó el entrenamiento
- Permite detectar problemas como sobreajuste o subajuste
- Ayuda a entender el comportamiento del modelo

**¿Cómo funciona?**

1. **Accuracy**: Muestra la precisión en entrenamiento vs validación
2. **Loss**: Muestra la pérdida en entrenamiento vs validación
3. **Dos gráficos**: Uno para cada métrica

**¿Qué patrones buscar?**

- **Sobreajuste**: Accuracy de entrenamiento sube, pero la de validación baja
- **Subajuste**: Ambas curvas se mantienen bajas
- **Buen entrenamiento**: Ambas curvas suben juntas
- **Early stopping efectivo**: El entrenamiento se detiene cuando la validación deja de mejorar

---

## 9. Función `main()`

```python
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
```

**¿Para qué sirve?**

- Orquesta todo el pipeline de machine learning
- Ejecuta las funciones en el orden correcto
- Genera todos los análisis y visualizaciones

**¿Cómo funciona?**

1. **Paso 1**: Carga y preprocesa los datos
2. **Paso 2**: Divide en conjuntos de entrenamiento y prueba
3. **Paso 3**: Crea la arquitectura de la red neuronal
4. **Paso 4**: Entrena el modelo
5. **Paso 5**: Evalúa el rendimiento
6. **Paso 6**: Genera visualizaciones

**¿Por qué esta estructura?**

- **Modular**: Cada función tiene una responsabilidad específica
- **Reutilizable**: Las funciones se pueden usar independientemente
- **Mantenible**: Fácil de modificar y depurar
- **Legible**: El flujo es claro y fácil de seguir

---

## ¿Cómo Verificar que el Código Funciona Correctamente?

### 1. **Indicadores de Ejecución Exitosa**

**Durante la ejecución, deberías ver:**

```
============================================================
RED NEURONAL PARA CLASIFICACIÓN DE FRUTAS MADURAS
============================================================
Cargando dataset de frutas...
Dataset cargado: 1000 muestras, 3 características
Dividiendo datos en entrenamiento (75%) y prueba (25%)...
Conjunto de entrenamiento: 750 muestras
Conjunto de prueba: 250 muestras
Creando red neuronal multicapa...
Arquitectura del modelo:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
capa_oculta (Dense)          (None, 10)                30
capa_salida (Dense)          (None, 1)                 11
=================================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
_________________________________________________________________
Entrenando modelo...
Epoch 1/100
94/94 [==============================] - 1s 4ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
...
Epoch X/100
94/94 [==============================] - 0s 3ms/step - loss: 0.XXXX - accuracy: 0.XXXX - val_loss: 0.XXXX - val_accuracy: 0.XXXX

Evaluando modelo...
Accuracy: 0.XXXX

Matriz de confusión:
[[XXX  XX]
 [ XX XXX]]

Reporte de clasificación:
              precision    recall  f1-score   support

   No Madura       0.XX      0.XX      0.XX        XXX
       Madura       0.XX      0.XX      0.XX        XXX

    accuracy                           0.XX        XXX
   macro avg       0.XX      0.XX      0.XX        XXX
weighted avg       0.XX      0.XX      0.XX        XXX

Generando gráficos...
[Se abren 4 ventanas con gráficos]

============================================================
ANÁLISIS COMPLETADO
Accuracy final: 0.XXXX
============================================================
```

### 2. **Métricas de Calidad Esperadas**

**Accuracy esperado:**

- **> 0.85**: Excelente rendimiento
- **0.75 - 0.85**: Buen rendimiento
- **0.65 - 0.75**: Rendimiento aceptable
- **< 0.65**: Rendimiento pobre

**Matriz de confusión balanceada:**

```
[[Verdadero Negativo    Falso Positivo ]
 [Falso Negativo       Verdadero Positivo]]
```

**Características de un buen entrenamiento:**

- Accuracy de entrenamiento y validación suben juntas
- No hay gap grande entre entrenamiento y validación
- Early stopping detiene el entrenamiento automáticamente
- Loss disminuye de manera estable

### 3. **Verificación Visual**

**Gráfico de distribución de clases:**

- Debe mostrar barras con números similares para ambas clases
- Colores diferentes para "No Madura" y "Madura"

**Gráfico de frontera de decisión:**

- Regiones coloreadas que separan las clases
- Puntos de datos superpuestos con colores según clase real
- Frontera suave y lógica

**Historial de entrenamiento:**

- Curvas de accuracy y loss que mejoran con el tiempo
- Curvas de entrenamiento y validación cercanas entre sí

### 4. **Señales de Problemas**

**Si algo no funciona bien:**

❌ **Error de archivo no encontrado:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'frutas_sinteticas_1000.csv'
```

**Solución**: Verificar que el archivo CSV esté en la misma carpeta

❌ **Accuracy muy bajo (< 0.6):**

- Posibles causas: Datos mal balanceados, modelo muy simple, datos de mala calidad
- **Solución**: Revisar el dataset, aumentar la complejidad del modelo

❌ **Sobreajuste (accuracy entrenamiento >> accuracy validación):**

- Posibles causas: Modelo muy complejo, pocos datos
- **Solución**: Reducir complejidad, agregar regularización, más datos

❌ **Subajuste (ambas accuracys bajas):**

- Posibles causas: Modelo muy simple, datos insuficientes
- **Solución**: Aumentar complejidad del modelo, mejorar características

### 5. **Pruebas Adicionales**

**Para verificar que todo funciona:**

1. **Ejecutar múltiples veces:**

   ```bash
   python red_neuronal_frutas.py
   python red_neuronal_frutas.py
   python red_neuronal_frutas.py
   ```

   - Los resultados deben ser similares (random_state=42)

2. **Verificar archivos generados:**

   - El script debe ejecutarse sin errores
   - Los gráficos deben abrirse automáticamente
   - No debe haber mensajes de error en consola

3. **Probar con datos nuevos:**
   - Crear frutas sintéticas con valores extremos
   - Verificar que las predicciones sean razonables
   - La frontera de decisión debe ser lógica

### 6. **Interpretación de Resultados**

**Un modelo exitoso debería:**

- ✅ Accuracy > 0.8 en datos de prueba
- ✅ Matriz de confusión balanceada
- ✅ Frontera de decisión lógica y suave
- ✅ Curvas de entrenamiento estables
- ✅ Early stopping funcionando
- ✅ Gráficos generándose correctamente

**Si cumple estos criterios, el código está funcionando correctamente y la red neuronal está aprendiendo efectivamente a clasificar frutas maduras de no maduras.**
