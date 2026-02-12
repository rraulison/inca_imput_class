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

- `experiment.n_sample: 10000`
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

## O que cada etapa faz

### 1) `prepare` (`src/data_preparation.py`)

- Filtra por ano (`DATAPRICON`, 2013-2023)
- Converte codigos "sem informacao" para `NaN` usando `config/dicionario_valores_validos.json`
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
- Shape final amostrado: `10,000 x 20`
- Classes finais (encoded):
  - `0->0`, `1->1`, `2->2`, `3->3`, `4->4`, `88->5`

Distribuicao de classes na amostra final (`y_prepared`):

| target | count |
|---:|---:|
| 0 | 617 |
| 1 | 2262 |
| 2 | 1013 |
| 3 | 841 |
| 4 | 1915 |
| 5 | 3352 |

### Cobertura experimental

- Tentativas totais em `all_results.csv`: `90`
- Execucoes validas com metricas: `72`
- Execucoes com erro: `18`
- Motivo dos erros: `cuML_MLP` indisponivel (`cupy/cuml MLPClassifier unavailable.`)

### Top combinacoes (ordenado por `f1_weighted_mean`)

| imputer | classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|:---|---:|---:|---:|---:|
| Media | XGBoost | 0.6432 | 0.6524 | 0.8972 | 92.7 |
| Mediana | XGBoost | 0.6427 | 0.6519 | 0.8972 | 92.7 |
| MissForest | XGBoost | 0.6404 | 0.6491 | 0.8945 | 124.4 |
| MICE_XGBoost | XGBoost | 0.6393 | 0.6484 | 0.8953 | 109.1 |
| MICE | XGBoost | 0.6383 | 0.6474 | 0.8959 | 94.1 |
| kNN | XGBoost | 0.6361 | 0.6454 | 0.8952 | 99.0 |

### Media por classificador (sobre todos os imputadores)

| classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|---:|---:|---:|---:|
| XGBoost | 0.6400 | 0.6491 | 0.8959 | 102.0 |
| CatBoost | 0.6159 | 0.6274 | 0.8832 | 267.2 |
| cuML_RF | 0.6222 | 0.6132 | 0.8779 | 35.8 |
| cuML_SVM | 0.5173 | 0.4800 | 0.7807 | 37.7 |

### Testes estatisticos

Friedman (global):

- `f1_weighted`: estatistica `66.2435`, `p=4.56e-06` (significativo)
- `auc_weighted`: estatistica `67.5060`, `p=2.93e-06` (significativo)

Post-hoc Wilcoxon pareado com Bonferroni:

- Nenhuma comparacao par-a-par ficou significativa (`p_bonf < 0.05`) nos arquivos atuais.
- Contexto: ha apenas 3 folds externos, o que reduz poder estatistico no post-hoc.

### Relatorio por classe do melhor modelo (Media + XGBoost)

| Classe | Precision | Recall | F1-Score | Support |
|:---|:---|:---|:---|---:|
| 0 | 0.8012 +- 0.0266 | 0.8719 +- 0.0124 | 0.8349 +- 0.0182 | 206 |
| 1 | 0.6360 +- 0.0200 | 0.6516 +- 0.0072 | 0.6437 +- 0.0138 | 754 |
| 2 | 0.3588 +- 0.0041 | 0.4551 +- 0.0143 | 0.4011 +- 0.0065 | 338 |
| 3 | 0.3064 +- 0.0073 | 0.3734 +- 0.0204 | 0.3364 +- 0.0109 | 280 |
| 4 | 0.6392 +- 0.0103 | 0.6099 +- 0.0037 | 0.6241 +- 0.0041 | 638 |
| 88 | 0.8629 +- 0.0131 | 0.7390 +- 0.0048 | 0.7961 +- 0.0081 | 1117 |
| macro avg | 0.6007 +- 0.0048 | 0.6168 +- 0.0027 | 0.6061 +- 0.0041 | 3333 |
| weighted avg | 0.6671 +- 0.0003 | 0.6432 +- 0.0016 | 0.6524 +- 0.0012 | 3333 |

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
- `missing_report_pre_filter.csv` e `missing_report_post_filter.csv`

Resultados brutos (`results/raw/`):

- `all_results.csv`
- `all_results_detailed.json`
- `checkpoint_classification.json`
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

- `cuML_MLP` nao esta disponivel no ambiente desta execucao.
- Com `n_outer_folds=3`, os testes post-hoc tem baixo poder estatistico.
- A qualidade final depende diretamente da qualidade do dicionario de codigos e da consistencia da base de origem.

## Proximos passos sugeridos

- Aumentar `n_outer_folds` (ex.: 5) para fortalecer inferencia estatistica.
- Testar calibracao de probabilidades para AUC multiclasses.
- Executar analises estratificadas por subgrupos clinicos quando aplicavel.
