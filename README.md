# SENTRIA Predictive Maintenance

> Capa de detección temprana de fallas sobre el despliegue SENTRIA existente.
> Sistema de anomalía y predicción de fallo en líneas de producción industrial,
> en modo **read-only** (recomendación, no control closed-loop).

**Status:** Specification phase
**Target delivery:** Q3 2026
**Cliente piloto:** Absara (continuación del despliegue SENTRIA actual)
**Maintainer:** Santiago Álvarez (sanalvcham@gmail.com)

---

## 1. Motivación

El sistema SENTRIA ya desplegado en planta de Absara (1,700 empleados, 31 líneas
de producción, 80+ usuarios activos) registra incidencias en tiempo real con
ruteo y escalación multi-agente. La capa de **predictive maintenance** explota
ese histórico estructurado de incidencias junto con telemetría de máquina y
variables ambientales para anticipar fallos antes de que ocurran.

El objetivo es producir alertas con **lead-time accionable** sobre el operador
responsable, sin tomar control de la máquina. La decisión de operar en modo
recomendación (no closed-loop) responde a contención de riesgo operacional y a
una etapa inicial de validación de calibración del modelo en producción.

## 2. Alcance del piloto

- **Líneas en scope:** 3 líneas de producción seleccionadas por su histórico de
  fallos documentados y disponibilidad de telemetría continua.
- **Horizonte de predicción:** ventana de alerta de 30 a 180 minutos antes del
  evento de fallo (lead-time-before-failure).
- **Modo de operación:** read-only. Las alertas se entregan al operador
  responsable; no hay shutdown automatizado.
- **Fuera de scope:** control closed-loop, optimización de mantenimiento
  preventivo, scheduling de paros programados.

## 3. Arquitectura propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                     Fuentes de datos                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Telemetría de máquina (sensores: temperatura, vibración,     │
│    consumo eléctrico, presión)                                  │
│ 2. Frecuencia y tipo de incidencia por línea (DynamoDB SENTRIA) │
│ 3. Variables ambientales (temp/humedad de planta, turno, lote)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Ingesta y feature engineering                      │
│  AWS Lambda + DynamoDB Streams → S3 (parquet, particionado      │
│  por línea+día)                                                 │
│  Features: ventanas deslizantes 5/15/60 min, agregaciones       │
│  estadísticas, lag features, ratios cruzados                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Ensamble de modelos                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Isolation       │  │ LSTM            │  │ ARIMA           │ │
│  │ Forest          │  │ Autoencoder     │  │ (baseline)      │ │
│  │ (anomalía       │  │ (reconstrucción │  │                 │ │
│  │ multivariada)   │  │ + residuo)      │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           Voting / score-level fusion ponderado                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Alerta y delivery (read-only)                      │
│  Score > umbral calibrado → Lambda → notificación WhatsApp      │
│  al operador (mismo canal SENTRIA actual)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Metodología de evaluación

### 4.1 Métricas primarias

| Métrica | Definición | Target inicial |
|---|---|---|
| Precision @ K | TP / (TP + FP) entre las K alertas más altas por turno | ≥ 0.60 |
| Recall | TP / (TP + FN) sobre fallos documentados | ≥ 0.50 |
| F1 | Media armónica de precision y recall | ≥ 0.55 |
| Lead-time mediano | Mediana de minutos entre alerta correcta y evento de fallo | ≥ 30 min |
| Tasa de falsa alarma | FP / turno / línea | ≤ 1.0 |

### 4.2 Validación

- **Hold-out temporal:** partición train/validation/test por bloques de tiempo
  consecutivos (no random) para preservar dependencia temporal.
- **Walk-forward evaluation:** evaluación incremental simulando despliegue real.
- **Comparación contra baseline:** umbral simple de SPC (control estadístico
  de proceso) sobre telemetría univariada.

### 4.3 Ablations planeadas

1. Cada modelo del ensamble en aislado (Isolation Forest solo, LSTM-AE solo,
   ARIMA solo) vs ensamble completo.
2. Con y sin variables ambientales para medir su contribución marginal.
3. Con y sin la señal de frecuencia de incidencia histórica (transferencia de
   conocimiento desde SENTRIA core).
4. Sensibilidad a la ventana de feature engineering (5 vs 15 vs 60 min).

### 4.4 Limitaciones documentadas

- **Label noise:** los fallos vienen de reportes manuales de operador; existe
  subreporting de fallos menores y posible mis-tagging de causa raíz.
- **Censoring:** intervenciones preventivas no escaladas a incidencia formal
  no aparecen en el dataset y pueden estar enmascarando señales útiles.
- **Drift:** condiciones de planta (cambio de SKU, mantenimientos
  programados, estacionalidad) introducen distribución shift que requerirá
  re-entrenamiento periódico documentado.
- **Generalización:** modelos entrenados en 3 líneas; transferencia a otras
  líneas/plantas requiere validación independiente.

## 5. Stack

- **Compute:** AWS Lambda (ingesta), SageMaker (entrenamiento y endpoint).
- **Almacenamiento:** S3 (parquet histórico), DynamoDB (alertas y feedback).
- **Modelado:** Python 3.11, scikit-learn (Isolation Forest), PyTorch
  (LSTM Autoencoder), statsmodels (ARIMA), Optuna (TPE para tuning).
- **Observabilidad:** CloudWatch (logs y métricas), modelo Evidently AI para
  drift monitoring.
- **Delivery:** Twilio / WhatsApp Business API (reutiliza pipeline SENTRIA).

## 6. Plan de trabajo (Q3 2026)

| Semana | Hito |
|---|---|
| W1-W2 | Acuerdo de datos con cliente, instrumentación de telemetría faltante, definición de etiquetas históricas con QA del cliente |
| W3-W4 | ETL y feature engineering en parquet, EDA, reporte de calidad de datos |
| W5-W6 | Baseline SPC + ARIMA, evaluación con walk-forward |
| W7-W8 | Isolation Forest y LSTM-AE, ablations, fusion ponderada |
| W9-W10 | Calibración de umbrales por línea, simulación de alertas en datos retenidos |
| W11-W12 | Despliegue read-only shadow (sin notificar a operadores), validación de tasa de falsa alarma |
| W13+ | Go-live escalonado con un operador piloto por línea |

## 7. Entregables

1. **Model card** documentando arquitectura, métricas, ablations, casos donde
   el modelo falla y supuestos operacionales.
2. **Reporte técnico** (~12-20 páginas) con introducción, trabajo relacionado,
   metodología, resultados, limitaciones y trabajo futuro. Formato compatible
   con preprint en arXiv o submission a workshop industrial.
3. **Dashboard de monitoreo** para el cliente: precisión rolling, tasa de
   falsa alarma, drift indicators.
4. **Repositorio reproducible** con código modular, tests y notebooks de
   ablations.

## 8. Posicionamiento de investigación

El proyecto se conecta con líneas activas en la academia y la industria:

- **Anomaly detection en sistemas industriales con label noise** — área
  abierta en literatura de PdM (predictive maintenance).
- **Ensembles heterogéneos para series de tiempo multivariadas** —
  evaluación comparativa contra modelos unitarios.
- **Causal vs correlational signals en mantenimiento predictivo** — papel
  de las variables ambientales y de proceso en alertas accionables.

Líneas de investigación en sistemas inteligentes que tocan este problema:
fusión de datos heterogéneos, razonamiento bajo incertidumbre en sistemas de
monitoreo, evaluación de utilidad downstream de modelos de detección.

## 9. Referencias relevantes (selección inicial)

- Pang, G. et al. *Deep Learning for Anomaly Detection: A Review.* ACM
  Computing Surveys, 2021.
- Hundman, K. et al. *Detecting Spacecraft Anomalies Using LSTMs and
  Nonparametric Dynamic Thresholding.* KDD 2018.
- Liu, F. T., Ting, K. M., Zhou, Z. H. *Isolation Forest.* ICDM 2008.
- Box, G. E. P., Jenkins, G. M. *Time Series Analysis: Forecasting and
  Control.* 5th ed.
- Saxena, A. et al. *Damage propagation modeling for aircraft engine
  run-to-failure simulation.* PHM 2008. (Dataset NASA C-MAPSS, referencia
  metodológica.)

---

## Estado del repositorio

Este repositorio contiene la **especificación técnica** del proyecto. El
código y los notebooks se añadirán durante la ejecución del piloto a partir
de Q3 2026. Issues y discusión técnica abiertas para colaboradores
autorizados.
