# 🎯 Modelo Activación 5tx en 30 días - SPIN

> Predicción de usuarios que completarán 5+ transacciones en los primeros 30 días post-signup

[![Model Version](https://img.shields.io/badge/version-1.3.0--lgbm-blue)]()
[![Python](https://img.shields.io/badge/python-3.10+-green)]()
[![LightGBM](https://img.shields.io/badge/model-LightGBM-orange)]()

---

## 📋 Descripción

Este notebook entrena un modelo de clasificación para predecir la probabilidad de que un usuario nuevo complete **5 o más transacciones** dentro de los primeros **30 días** después de su registro (signup).

El modelo es utilizado por los equipos de **Marketing y Growth** para:
- Orquestar campañas de nudges personalizados
- Priorizar push notifications y SMS
- Segmentar usuarios por propensión de activación

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE ML - 5TX 30D                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   BigQuery   │───▶│  FeatureBuilder │───▶│  LightGBM  │───▶ Scores  │
│  │   (raw data) │    │  .transform()   │    │  .predict() │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
│  Columnas entrada:        Features generadas:      Output:              │
│  - user_id                - age_years              - p_5tx_30d          │
│  - signup_ts              - gender_bin             - user_id            │
│  - gender                 - state_OHE (32)                              │
│  - stateName              - signup_dow/month                            │
│  - birth_date             - near_payday_*                               │
│  - channelDetail          - is_holiday_mx                               │
│  - ...                    - ...                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos

```
├── model_5_trx_1_3_0-MLOps_JFAA.ipynb   # Notebook principal
├── README.md                             # Este archivo
├── artifacts/                            # Artefactos serializados
│   ├── model_5tx_lgbm.joblib            # Modelo LightGBM entrenado
│   ├── feature_builder.joblib           # Transformador de features
│   └── metadata.json                    # Metadata y métricas
├── inference_lgbm.py                    # Script de inferencia
└── notebook_section_lgbm.py             # Código del modelo (snippet)
```

---

## ⚙️ Configuración

```python
@dataclass
class Config:
    project_id: str = "spin-aip-singularity-comp-sb"
    label_col: str = "label_5tx_30d"
    signup_ts_col: str = "signup_ts"
    tz_local: str = "America/Mexico_City"
    embargo_days: int = 3       # Gap entre train y holdout
    holdout_days: int = 14      # Días para evaluación temporal
    random_state: int = 42
```

---

## 🧠 Modelo

### LightGBM Classifier

```python
LGBMClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
```

### Métricas Esperadas

| Métrica | Validación | Holdout |
|---------|------------|---------|
| Average Precision | ~0.98 | ~0.98 |
| ROC-AUC | ~0.96 | ~0.96 |
| Brier Score | ~0.08 | ~0.08 |

---

## 🔧 Features

### Categóricas (transformadas)

| Original | Transformada | Valores |
|----------|--------------|---------|
| `gender` | `gender_bin` | 0=male, 1=female |
| `user_type` | `user_type_tri` | 0=HYBRID, 1=DIGITAL, 2=ANALOG |
| `channelDetail` | `channel_detail_code` | 0-5 |
| `stateName` | `state_*` (OHE) | 32 estados |
| `birthState` | `birth_bucket` | 0-5 (regiones) |

### Temporales (generadas)

| Feature | Descripción |
|---------|-------------|
| `signup_dow` | Día de la semana (0-6) |
| `signup_month` | Mes del año (1-12) |
| `signup_week` | Semana ISO |
| `signup_daypart` | 0=mañana, 1=tarde, 2=noche |
| `is_holiday_mx` | Día festivo en México |
| `near_payday_*` | Cercanía a día de pago (1, 15, fin de mes) |

### Otras

| Feature | Descripción |
|---------|-------------|
| `age_years` | Edad calculada al signup |
| `card_linked_before_signup` | Tarjeta vinculada antes |
| `card_linked_lag_days` | Días desde vinculación |
| `phn_confir` / `email_confir` | Confirmaciones |
| `accountLevel` | Nivel de cuenta |
| `has_premia` | Tiene cuenta Premia |

---

## ⚠️ Anti-Leakage

### Columnas PROHIBIDAS en features

```python
LEAKY_ALWAYS = {
    "label_5tx_30d",           # Target
    "label_activated_30d",     # Label alternativo
    "tx_30d_count",            # Info post-signup
    "tx_30d_amount",           # Info post-signup
    "first_tx_type",           # Info post-activación
    "first_tx_amount",         # Info post-activación
    "activation_date_30d",     # Fecha de activación
    "days_to_first_activation" # Calculado post-facto
}
```

### Validación Temporal

```
Timeline:
─────────────────────────────────────────────────────────▶
│◄──────── TRAIN ────────►│◄─ EMBARGO ─►│◄── HOLDOUT ──►│
                          │    (3 días)  │   (14 días)   │
                          train_end    holdout_start   max_date
```

---

## 🚀 Inferencia

### Patrón Minimalista (3 líneas)

```python
import joblib
import pandas as pd

# 1. Cargar datos nuevos
data = pd.read_parquet("nuevos_usuarios.parquet")

# 2. Transformar
fb = joblib.load("artifacts/feature_builder.joblib")
X = fb.transform(data).X

# 3. Predecir
model = joblib.load("artifacts/model_5tx_lgbm.joblib")
scores = model.predict_proba(X)[:, 1]
```

### Uso en Producción (BigQuery)

```python
from google.cloud import bigquery
import joblib
import pandas as pd

# Cargar artefactos (una vez al inicio)
model = joblib.load("gs://bucket/artifacts/model_5tx_lgbm.joblib")
fb = joblib.load("gs://bucket/artifacts/feature_builder.joblib")

# Query nuevos usuarios
client = bigquery.Client()
query = """
    SELECT * FROM `project.dataset.nuevos_usuarios`
    WHERE signup_date = CURRENT_DATE() - 1
"""
data = client.query(query).to_dataframe()

# Scoring
X = fb.transform(data).X
scores = model.predict_proba(X)[:, 1]

# Guardar scores
result = pd.DataFrame({
    "user_id": data["user_id"],
    "p_5tx_30d": scores,
    "score_date": pd.Timestamp.now()
})
result.to_gbq("project.dataset.scores_5tx", if_exists="append")
```

---

## 📦 Requisitos

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
joblib
holidays
google-cloud-bigquery
```

### Instalación

```bash
pip install pandas numpy scikit-learn lightgbm joblib holidays
```

---

## 📊 Datos de Entrada

### Tabla BigQuery

```
spin-aip-singularity-comp-sb.model_activation.dataste_model_activation_timewindow_30D_V-1-5-0
```

### Columnas Requeridas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `user_id` | STRING | ID único del usuario |
| `signup_ts` | TIMESTAMP | Timestamp de registro |
| `signup_date` | DATE | Fecha de registro |
| `gender` | STRING | "male" / "female" |
| `user_type` | STRING | "HYBRID" / "DIGITAL" / "ANALOG" |
| `channelDetail` | STRING | Canal de adquisición |
| `stateName` | STRING | Estado (siglas) |
| `birthState` | STRING | Estado de nacimiento |
| `birth_date` | DATE | Fecha de nacimiento |
| `accountLevel` | INT | Nivel de cuenta (1-3) |
| `has_premia` | INT | 0/1 |

---

## 🔄 Versionado

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.3.0-lgbm | 2025-12 | Migración a LightGBM, anti-leakage reforzado |
| 1.2.0 | 2025-11 | HistGradientBoosting, RobustScaler |
| 1.1.0 | 2025-10 | Feature engineering temporal |
| 1.0.0 | 2025-09 | Versión inicial |

---

## 👥 Contacto

- **Owner**: Fernando Aguilar
- **Proyecto**: SPIN 5 trx in 30d

---

## 📝 Licencia

Uso interno - SPIN
