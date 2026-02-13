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

Modos de runtime (via argparse):

```bash
# default: tuning completo
python main.py --runtime-mode default

# hybrid: tuning leve em subset + refit no treino completo
python main.py --runtime-mode hybrid --n-sample 100000 --tune-max-samples 20000

# fast: sem tuning (usa hiperparametros fixos) para rodar 100k rapido
python main.py --runtime-mode fast --n-sample 100000
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
- Shape final amostrado: `100,000 x 26`
- Classes finais (encoded): `0->0`, `1->1`, `2->2`, `3->3`, `4->4`, `88->5`

Distribuicao de classes na amostra final (`y_prepared`):

| target | count |
|---:|---:|
| 0 | 6166 |
| 1 | 22625 |
| 2 | 10128 |
| 3 | 8413 |
| 4 | 19147 |
| 5 | 33521 |

### Cobertura experimental

- Runtime desta rodada: `hybrid` (`n_sample=100000`, `tune_max_samples=20000`)
- Combinacoes avaliadas em `summary.csv`: `18` (`6` imputadores x `3` classificadores)
- Folds esperados: `54` (`18 x 3`)
- Folds validos com metricas: `53`
- Falhas: `1` fold (`MICE_XGBoost + cuML_SVM`) por `CUDA out_of_memory`

### Top combinacoes (ordenado por `f1_weighted_mean`)

| imputer | classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|:---|---:|---:|---:|---:|
| Media | XGBoost | 0.6782 | 0.6878 | 0.9265 | 38.2 |
| kNN | XGBoost | 0.6776 | 0.6871 | 0.9262 | 252.7 |
| Mediana | XGBoost | 0.6750 | 0.6855 | 0.9254 | 37.7 |
| MICE_XGBoost | XGBoost | 0.6741 | 0.6847 | 0.9252 | 88.0 |
| MissForest | XGBoost | 0.6735 | 0.6841 | 0.9250 | 133.1 |
| MICE | XGBoost | 0.6709 | 0.6823 | 0.9241 | 51.7 |

### Media por classificador (sobre todos os imputadores)

| classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
|:---|---:|---:|---:|---:|
| XGBoost | 0.6749 | 0.6853 | 0.9254 | 100.2 |
| cuML_RF | 0.6668 | 0.6583 | 0.9090 | 67.1 |
| cuML_SVM | 0.5303 | 0.5114 | 0.7763 | 87.1 |

### Testes estatisticos

Friedman (global, 18 combinacoes):

- `f1_weighted`: estatistica `46.7697`, `p=7.40e-05` (significativo)
- `auc_weighted`: estatistica `47.2023`, `p=6.33e-05` (significativo)

Post-hoc Wilcoxon pareado com Bonferroni:

- Nenhuma comparacao par-a-par ficou significativa (`p_bonf < 0.05`) nos arquivos atuais.
- Contexto: ha apenas 3 folds externos, o que reduz poder estatistico no post-hoc.

### Relatorio por classe do melhor modelo (Media + XGBoost)

| Classe | Precision | Recall | F1-Score | Support |
|:---|:---|:---|:---|---:|
| 0 | 0.5164 +- 0.0126 | 0.6453 +- 0.0249 | 0.5731 +- 0.0024 | 2055 |
| 1 | 0.6990 +- 0.0050 | 0.6305 +- 0.0140 | 0.6628 +- 0.0057 | 7542 |
| 2 | 0.4119 +- 0.0028 | 0.5050 +- 0.0194 | 0.4536 +- 0.0094 | 3376 |
| 3 | 0.3625 +- 0.0061 | 0.4719 +- 0.0108 | 0.4098 +- 0.0005 | 2804 |
| 4 | 0.6827 +- 0.0037 | 0.6446 +- 0.0101 | 0.6630 +- 0.0048 | 6382 |
| 88 | 0.9255 +- 0.0027 | 0.8397 +- 0.0088 | 0.8805 +- 0.0050 | 11174 |
| macro avg | 0.5996 +- 0.0010 | 0.6228 +- 0.0038 | 0.6072 +- 0.0001 | 33333 |
| weighted avg | 0.7031 +- 0.0014 | 0.6782 +- 0.0032 | 0.6878 +- 0.0021 | 33333 |

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

- Nesta rodada (`hybrid`, `n=100000`), houve 1 falha por memoria GPU (`CUDA out_of_memory`) em `MICE_XGBoost + cuML_SVM`.
- Com `n_outer_folds=3`, os testes post-hoc tem baixo poder estatistico.
- A qualidade final depende diretamente da qualidade do dicionario de codigos e da consistencia da base de origem.

## Proximos passos sugeridos

- Aumentar `n_outer_folds` (ex.: 5) para fortalecer inferencia estatistica.
- Testar calibracao de probabilidades para AUC multiclasses.
- Executar analises estratificadas por subgrupos clinicos quando aplicavel.
