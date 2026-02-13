# Pipeline de Imputacao e Classificacao para Estadiamento Oncologico (SisRHC/INCA)

Projeto para comparar metodos de imputacao de dados faltantes e classificadores multiclasse no problema de estadiamento oncologico (`ESTADIAM`) com base em dados do SisRHC/INCA.

## Visao geral

Este repositorio implementa um pipeline completo em 4 etapas:

1. `prepare`: limpeza, filtro temporal, tratamento de codigos "sem informacao", encoding e amostragem estratificada.
2. `impute`: imputacao fold-a-fold (treino/teste) sem vazamento de dados.
3. `classify`: classificacao com validacao cruzada aninhada e tuning de hiperparametros.
4. `analyze`: consolidacao de metricas, testes estatisticos, tabelas e figuras.

## Objetivo

Avaliar combinacoes de:

- Imputadores: `Media`, `Mediana`, `kNN`, `MICE`, `MICE_XGBoost`, `MissForest`
- Classificadores: `XGBoost`, `CatBoost`, `cuML_RF`, `cuML_SVM`, `cuML_MLP`

sobre as metricas:

- `accuracy`
- `recall_weighted`
- `f1_weighted` (metrica principal de tuning)
- `auc_weighted`
- metricas macro auxiliares

## Estrutura do projeto

```text
.
|-- config/
|   |-- config.yaml
|   `-- dicionario_valores_validos.json
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- imputed/
|-- results/
|   |-- raw/
|   |-- tables/
|   `-- figures/
|-- src/
|   |-- config_loader.py
|   |-- data_preparation.py
|   |-- run_imputation.py
|   |-- run_classification.py
|   `-- run_analysis.py
`-- main.py
```

## Requisitos

- Python `>=3.10` (ambiente atual executado com Python 3.13)
- Dependencias em `requirements.txt`
- GPU CUDA recomendada para modelos `cuML_*` e aceleracao do XGBoost/CatBoost

Pacotes principais:

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `xgboost`, `catboost`
- `matplotlib`, `seaborn`
- `cupy`, `cudf`, `cuml` (opcionais, para GPU)

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se quiser executar em CPU apenas, ajuste no `config/config.yaml`:

```yaml
hardware:
  use_gpu: false
```

## Configuracao (`config/config.yaml`)

Parametros mais importantes:

- `experiment.n_sample: 100000`
- `data.filepath: data/raw/df_raw.parquet`
- `data.target_col: ESTADIAM`
- `data.valid_classes: [0,1,2,3,4,88]`
- `data.missing_threshold: 0.60`
- `cv.n_outer_folds: 3`
- `cv.n_inner_folds: 3`
- `classification.tuning.scoring: f1_weighted`

## Como executar

Pipeline completo:

```bash
python main.py
# ou
python main.py --step all
```

Executar por etapa:

```bash
python main.py --step prepare
python main.py --step impute
python main.py --step classify
python main.py --step analyze
```

Executar subconjuntos:

```bash
python main.py --step impute --imputer MICE_XGBoost
python main.py --step classify --classifier XGBoost
python main.py --step classify --imputer Media --classifier CatBoost
```

Modos de runtime (via argparse):

```bash
# default: tuning completo
python main.py --runtime-mode default

# hybrid: tuning leve em subset + refit no treino completo
python main.py --runtime-mode hybrid --n-sample 1000000 --tune-max-samples 60000

# fast: sem tuning (usa hiperparametros fixos) para execucao rapida
python main.py --runtime-mode fast --n-sample 1000000
```

## O que cada etapa faz

### 1) `prepare` (`src/data_preparation.py`)

- Filtra por ano (`DATAPRICON`, 2013-2023)
- Converte codigos nao informativos para `NaN` (ex.: "Sem informacao", "Nao avaliado", "Nao se aplica", ocupacao ignorada) usando `config/dicionario_valores_validos.json`
- Filtra classes validas do alvo
- Remove variaveis de leakage/administrativas
- Reduz cardinalidade de variaveis de alta cardinalidade
- Aplica `LabelEncoder` nas categoricas
- Faz amostragem estratificada (`n_sample`)
- Salva:
  - `data/processed/X_prepared.parquet`
  - `data/processed/y_prepared.parquet`
  - `data/processed/encoders.pkl`
  - `results/tables/metadata.json`

### 2) `impute` (`src/run_imputation.py`)

- Cria folds estratificados externos
- Ajusta imputador apenas no treino de cada fold
- Transforma treino e teste separadamente
- Arredonda variaveis categoricas imputadas para valores validos
- Salva matrizes imputadas por fold em `data/imputed/`
- Salva tempos em `results/raw/tempos_imputacao.json`

### 3) `classify` (`src/run_classification.py`)

- Carrega folds imputados
- Escalona somente quando necessario (ex.: SVM/MLP)
- Executa tuning com CV interna (`RandomizedSearchCV` para sklearn; busca manual para cuML)
- Suporta `runtime-mode`:
  - `default`: tuning completo do `config`
  - `hybrid`: tuning em subset estratificado (mais rapido), com refit no treino completo
  - `fast`: sem tuning, com hiperparametros fixos predefinidos
- Calcula metricas por fold
- Gera checkpoint incremental (`results/raw/checkpoint_classification.json`)
- Salva:
  - `results/raw/all_results.csv`
  - `results/raw/all_results_detailed.json`

### 4) `analyze` (`src/run_analysis.py`)

- Agrega media e desvio por combinacao imputador+classificador
- Gera ranking
- Executa Friedman + Wilcoxon com correcao Bonferroni
- Produz tabelas `.csv` e `.tex`
- Gera figuras em PNG/PDF

## Resultados desta execucao

Baseados nos artefatos atuais em `results/`.

### Resumo dos dados apos preparo

- Shape original: `5,399,686 x 47`
- Apos filtro de data (2013-2023): `3,211,670`
- Apos filtro de classes alvo: `1,586,358`
- Shape final amostrado: `1,000,000 x 25`
- Classes finais (encoded): `0->0`, `1->1`, `2->2`, `3->3`, `4->4`, `88->5`

Distribuicao de classes na amostra final (`y_prepared`):

| target | count |
|---:|---:|
| 0 | 61659 |
| 1 | 226253 |
| 2 | 101279 |
| 3 | 84130 |
| 4 | 191467 |
| 5 | 335212 |

### Cobertura experimental

- Runtime desta rodada: `hybrid` (`n_sample=1000000`, `tune_max_samples=60000`)
- Combinacoes avaliadas em `summary.csv`: `20` (`18` principais + `2` baselines: `NoImpute + XGBoost` e `RawSemEncoding + CatBoost`)
- Folds esperados: `60` (`20 x 3`)
- Folds validos com metricas: `60`
- Falhas: `0`

### Top combinacoes (ordenado por `f1_weighted_mean`)

| imputer | classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|:---|---:|---:|---:|---:|
| Media | XGBoost | 0.6941 | 0.7060 | 0.9394 | 62.2 |
| NoImpute | XGBoost | 0.6941 | 0.7059 | 0.9394 | 62.6 |
| Mediana | XGBoost | 0.6940 | 0.7058 | 0.9392 | 58.7 |
| MICE | XGBoost | 0.6938 | 0.7056 | 0.9390 | 127.5 |
| MICE_XGBoost | XGBoost | 0.6933 | 0.7051 | 0.9389 | 271.9 |
| kNN | XGBoost | 0.6931 | 0.7049 | 0.9387 | 2194.0 |

### Media por classificador (sobre todos os imputadores)

| classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|---:|---:|---:|---:|
| XGBoost | 0.6936 | 0.7054 | 0.9390 | 506.9 |
| CatBoost | 0.6795 | 0.6923 | 0.9336 | 121.1 |
| cuML_RF | 0.6869 | 0.6791 | 0.9206 | 529.8 |
| cuML_SVM | 0.5139 | 0.4950 | 0.7663 | 555.3 |

### Testes estatisticos

Friedman (global, 20 combinacoes):

- `f1_weighted`: estatistica `55.5894`, `p=1.89e-05` (significativo)
- `auc_weighted`: estatistica `56.1613`, `p=1.54e-05` (significativo)

Post-hoc Wilcoxon pareado com Bonferroni:

- Nenhuma comparacao par-a-par ficou significativa (`p_bonf < 0.05`) nos arquivos atuais.
- Contexto: ha apenas 3 folds externos, o que reduz poder estatistico no post-hoc.

### Relatorio por classe do melhor modelo (Media + XGBoost)

| Classe | Precision | Recall | F1-Score | Support |
|:---|:---|:---|:---|---:|
| 0 | 0.4940 +- 0.0114 | 0.7283 +- 0.0023 | 0.5886 +- 0.0074 | 20553 |
| 1 | 0.7452 +- 0.0014 | 0.6121 +- 0.0100 | 0.6721 +- 0.0063 | 75418 |
| 2 | 0.4316 +- 0.0056 | 0.5404 +- 0.0018 | 0.4799 +- 0.0039 | 33760 |
| 3 | 0.3752 +- 0.0079 | 0.5423 +- 0.0015 | 0.4434 +- 0.0053 | 28043 |
| 4 | 0.7210 +- 0.0053 | 0.6504 +- 0.0069 | 0.6839 +- 0.0061 | 63822 |
| 88 | 0.9465 +- 0.0013 | 0.8528 +- 0.0097 | 0.8972 +- 0.0059 | 111737 |
| macro avg | 0.6189 +- 0.0053 | 0.6544 +- 0.0041 | 0.6275 +- 0.0058 | 333333 |
| weighted avg | 0.7297 +- 0.0035 | 0.6941 +- 0.0067 | 0.7060 +- 0.0059 | 333333 |

## Figuras (geradas automaticamente)

### Missing por variavel

![Missing rates](results/figures/missing_rates.png)

### Heatmaps de metricas

![Heatmap Accuracy](results/figures/heatmap_accuracy.png)

![Heatmap F1](results/figures/heatmap_f1_weighted.png)

![Heatmap AUC](results/figures/heatmap_auc_weighted.png)

![Heatmap Recall](results/figures/heatmap_recall_weighted.png)

### Boxplots das metricas por metodo

![Boxplots](results/figures/boxplots_metrics.png)

### Matriz de confusao (melhor modelo)

![Confusion Matrix](results/figures/confusion_matrix_best.png)

### F1 por classe (melhor classificador por imputador)

![Per class F1](results/figures/per_class_f1.png)

### Comparacao radar

![Radar](results/figures/radar_best.png)

### Tempo por etapa

![Timing](results/figures/timing_stacked.png)

## Principais artefatos de saida

Tabelas (`results/tables/`):

- `summary.csv`
- `main_table.csv` e `main_table.tex`
- `ranking.csv`
- `per_class_report.csv`
- `stat_friedman_*.csv`
- `stat_wilcoxon_*.csv`
- `metadata.json`
- `missing_report_raw.csv`, `missing_report_pre_filter.csv` e `missing_report_post_filter.csv`

Resultados brutos (`results/raw/`):

- `all_results.csv`
- `all_results_detailed.json`
- `checkpoint_classification.json`
- Para modos nao-default: `all_results_<mode>.csv`, `all_results_detailed_<mode>.json`, `checkpoint_classification_<mode>.json`
- `tempos_imputacao.json`
- `experiment.log`
- `pip_freeze.txt`

## Reprodutibilidade

- Semente global: `42`
- O ambiente e versoes sao registrados em `results/raw/pip_freeze.txt` e `results/raw/experiment.log`
- Para reproducao, execute:

```bash
python main.py --step all --config config/config.yaml
```

## Limitacoes atuais

- Nesta rodada (`hybrid`, `n=1000000`), nao houve falhas de fold, mas metodos com imputacao pesada (principalmente `kNN` e `MissForest`) elevaram bastante o tempo total.
- Com `n_outer_folds=3`, os testes post-hoc tem baixo poder estatistico.
- A qualidade final depende diretamente da qualidade do dicionario de codigos e da consistencia da base de origem.

## Proximos passos sugeridos

- Aumentar `n_outer_folds` (ex.: 5) para fortalecer inferencia estatistica.
- Testar calibracao de probabilidades para AUC multiclasses.
- Executar analises estratificadas por subgrupos clinicos quando aplicavel.
