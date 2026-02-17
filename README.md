# Pipeline de Imputacao e Classificacao para Estadiamento Oncologico (SisRHC/INCA)

Projeto para comparar metodos de imputacao de dados faltantes e classificadores multiclasse no problema de estadiamento oncologico (`ESTADIAM`) com base em dados do SisRHC/INCA.

## Visao geral

Este repositorio implementa um pipeline completo em 4 etapas:

1. `prepare`: limpeza, filtro temporal, tratamento de codigos "sem informacao", encoding e amostragem estratificada.
2. `impute`: imputacao fold-a-fold (treino/teste) sem vazamento de dados.
3. `classify`: classificacao com validacao cruzada aninhada e tuning de hiperparametros.
4. `analyze`: consolidacao de metricas, testes estatisticos, tabelas e figuras.

Analises complementares (pos-pipeline) tambem estao disponiveis:

- `imputation_effect_stats`: inferencia pareada para quantificar o efeito da imputacao no desempenho.
- `ordinal_sensitivity`: metricas ordinais e sensibilidade com/sem classe `88`.

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
|   |-- metrics_utils.py      # shared metrics (compute_metrics, coerce_confusion_matrix, ...)
|   |-- stats_utils.py         # shared stats (bootstrap, wilcoxon, cohen_dz, ...)
|   |-- run_imputation.py
|   |-- run_classification.py
|   |-- run_analysis.py
|   |-- run_imputation_effect_stats.py
|   |-- run_ordinal_sensitivity.py
|   `-- run_tabicl.py
|-- tests/
|   `-- test_shared_utils.py
|-- main.py
`-- pytest.ini
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
# Recomendado: usar o ambiente conda rapids-25.10 (inclui cuML, cuDF, cuPy)
conda activate rapids-25.10

# Instalar dependencias adicionais
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** Os classificadores `cuML_RF` e `cuML_SVM` requerem o RAPIDS toolkit.
> Certifique-se de que o ambiente `rapids-25.10` (ou compativel) esta ativado
> antes de executar o pipeline. Sem ele, apenas `XGBoost` e `CatBoost` funcionarao.

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
- `cv.n_outer_folds: 5`
- `cv.n_inner_folds: 5`
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

Analises complementares (nao exigem rerodar `prepare/impute/classify` se os artefatos ja existem):

```bash
# efeito da imputacao na metrica principal
python src/run_imputation_effect_stats.py --metric f1_weighted

# sensibilidade ordinal com/sem classe 88
python src/run_ordinal_sensitivity.py
```

Parametros uteis:

```bash
# margem pratica de equivalencia de 0.5 p.p.
python src/run_imputation_effect_stats.py --metric f1_weighted --equivalence-margin 0.005

# trocar baseline e numero de bootstraps
python src/run_ordinal_sensitivity.py --baseline NoImpute --bootstrap-iters 5000
```

Teste do TabICL (script dedicado, sem alterar pipeline principal):

```bash
# Reusa os mesmos folds de data/imputed/fold_indices.json
python src/run_tabicl.py --input-source raw_prepared --split-source existing

# Rodar com menos linhas (teste rapido)
python src/run_tabicl.py --split-source new --max-rows 50000 --max-steps 80 --patience 10
```

Saidas:

- `results/raw/tabicl_results.csv`
- `results/raw/tabicl_results_detailed.json`

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
- Gera checkpoint incremental (`results/raw/checkpoint_classification_<mode>.json`)
- Salva:
  - `results/raw/all_results.csv`
  - `results/raw/all_results_detailed.json`

### 4) `analyze` (`src/run_analysis.py`)

- Agrega media e desvio por combinacao imputador+classificador
- Gera ranking
- Executa Friedman + Wilcoxon com correcao Bonferroni
- Produz tabelas `.csv` e `.tex`
- Gera figuras em PNG/PDF

### 5) `imputation_effect_stats` (`src/run_imputation_effect_stats.py`)

- Le `results/raw/all_results.csv` (nivel fold)
- Compara imputadores de forma pareada nos mesmos blocos (`classifier`, `fold`)
- Reporta:
  - `delta_mean`, `delta_median`, IC bootstrap
  - Wilcoxon pareado + correcao Holm
  - tamanho de efeito (`cohen_dz`, `rank_biserial`)
  - testes de equivalencia (TOST) e nao-inferioridade
- Salva em `results/tables/imputation_effect/`

### 6) `ordinal_sensitivity` (`src/run_ordinal_sensitivity.py`)

- Le `results/raw/all_results_detailed.json` e `results/tables/metadata.json`
- Calcula metricas ordinais por fold a partir da matriz de confusao:
  - `qwk` (quadratic weighted kappa)
  - `mae_distance`, `rmse_distance`
  - `severe_error_rate` (erro com distancia >= 2)
  - `within_one_rate`
- Executa dois cenarios:
  - `all_classes`
  - `without_88` (remove a classe codificada de `88`)
- Salva em `results/tables/ordinal_sensitivity/`

## Resultados desta execucao

Baseados nos artefatos atuais em `results/`.

### Resumo dos dados apos preparo

- Shape original: `5,399,686 x 47`
- Apos filtro de data (2013-2023): `3,211,670`
- Apos filtro de classes alvo: `1,586,358`
- Shape final amostrado: `1,586,358 x 25` (todos os registros validos)
- Classes finais (encoded): `0->0`, `1->1`, `2->2`, `3->3`, `4->4`, `88->5`

Distribuicao de classes na amostra final (`y_prepared`):

| target |  count |
| -----: | -----: |
|      0 |  97814 |
|      1 | 358918 |
|      2 | 160664 |
|      3 | 133461 |
|      4 | 303735 |
|      5 | 531766 |

### Cobertura experimental

- Runtime desta rodada: `hybrid` (`n_sample=999999999`, `tune_max_samples=20000`, `inner_folds=2`)
- Combinacoes avaliadas em `summary.csv`: `20` (`18` principais + `2` baselines: `NoImpute + XGBoost` e `RawSemEncoding + CatBoost`)
- Folds esperados: `100` (`20 x 5`)
- Folds validos com metricas: `100`
- Falhas: `0`

### Top combinacoes (ordenado por `f1_weighted_mean`)

| imputer      | classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
| :----------- | :--------- | ------------: | ---------------: | ----------------: | ------------------: |
| MICE_XGBoost | XGBoost    |        0.6903 |           0.7028 |            0.9380 |               426.6 |
| MissForest   | XGBoost    |        0.6902 |           0.7028 |            0.9380 |              1473.8 |
| Media        | XGBoost    |        0.6895 |           0.7021 |            0.9376 |                55.0 |
| Mediana      | XGBoost    |        0.6895 |           0.7021 |            0.9376 |                55.1 |
| kNN          | XGBoost    |        0.6893 |           0.7020 |            0.9377 |              3679.0 |
| NoImpute     | XGBoost    |        0.6873 |           0.7002 |            0.9368 |                53.3 |

### Media por classificador (sobre todos os imputadores)

| classifier | accuracy_mean | f1_weighted_mean | auc_weighted_mean | time_total_mean (s) |
| :--------- | ------------: | ---------------: | ----------------: | ------------------: |
| XGBoost    |        0.6888 |           0.7015 |            0.9374 |               845.4 |
| CatBoost   |        0.6817 |           0.6944 |            0.9343 |               119.8 |
| cuML_RF    |        0.6886 |           0.6815 |            0.9218 |               934.6 |
| cuML_SVM   |        0.5086 |           0.4841 |            0.7661 |               958.7 |

### Testes estatisticos

Friedman (global, 20 combinacoes):

- `f1_weighted`: estatistica `89.7168`, `p=3.72e-11` (significativo)
- `auc_weighted`: estatistica `89.9634`, `p=3.37e-11` (significativo)

Post-hoc Wilcoxon pareado com Bonferroni:

- Nenhuma comparacao par-a-par ficou significativa (`p_bonf < 0.05`) nos arquivos atuais.
- Menor `p_bonf` observado: `1.0` (F1 e AUC).

### Efeito da imputacao (`results/tables/imputation_effect/`)

Configuracao desta rodada (manifesto em `results/tables/imputation_effect/manifest_f1_weighted.json`):

- metrica: `f1_weighted`
- `alpha=0.05`
- margem de equivalencia: `0.005` (0.5 p.p.)
- bootstrap: `5000` iteracoes
- baseline: `NoImpute`

Resumo global:

- Comparacoes pareadas globais entre imputadores: `21`
- Menor `p_wilcoxon_holm`: `1.0` (nenhuma comparacao significativa apos Holm)
- Maior diferenca media absoluta entre imputadores (`delta_mean`): `0.0056` (~0.56 p.p.)

Comparacao contra baseline `NoImpute` (XGBoost, `n_pairs=5`):

| imputer      | delta_mean F1 vs NoImpute | IC95% bootstrap        | p_wilcoxon_holm | equivalente (TOST, margem=0.005) |
| :----------- | ------------------------: | :--------------------- | --------------: | :------------------------------: |
| Media        |                 +0.001903 | [-0.000923, +0.007034] |          1.0000 |               Nao                |
| Mediana      |                 +0.001898 | [-0.000927, +0.006971] |          1.0000 |               Nao                |
| MICE         |                 -0.001910 | [-0.002328, -0.001509] |          0.3750 |               Sim                |
| MICE_XGBoost |                 +0.002648 | [-0.001440, +0.010287] |          1.0000 |               Nao                |
| kNN          |                 +0.001762 | [-0.002672, +0.009839] |          1.0000 |               Nao                |
| MissForest   |                 +0.002561 | [-0.001791, +0.010483] |          1.0000 |               Nao                |

Leitura direta para o objetivo do projeto:

- Nesta execucao, nao houve evidencia de ganho estatisticamente robusto de imputacao sobre `NoImpute`.
- As diferencas observadas ficaram pequenas em magnitude pratica (ordem de milesimos de F1), com aumento de custo para imputadores mais pesados.

### Sensibilidade ordinal e classe 88 (`results/tables/ordinal_sensitivity/`)

Configuracao desta rodada (manifesto em `results/tables/ordinal_sensitivity/manifest_ordinal.json`):

- baseline: `NoImpute`
- `alpha=0.05`
- margem de equivalencia em QWK: `0.005`
- bootstrap: `5000` iteracoes
- mapeamento alvo: `88 -> 5`
- linhas fold-level avaliadas: `200`

Melhor QWK por cenario:

| scenario    | melhor combinacao        | qwk_mean |
| :---------- | :----------------------- | -------: |
| all_classes | `MICE_XGBoost + XGBoost` |   0.7808 |
| without_88  | `MICE_XGBoost + XGBoost` |   0.6719 |

Media por classificador (QWK), com e sem `88`:

| classifier | QWK all_classes | QWK without_88 | delta (without_88 - all_classes) |
| :--------- | --------------: | -------------: | -------------------------------: |
| XGBoost    |          0.7787 |         0.6696 |                          -0.1091 |
| CatBoost   |          0.7704 |         0.6575 |                          -0.1129 |
| cuML_RF    |          0.7451 |         0.6659 |                          -0.0791 |
| cuML_SVM   |          0.4826 |         0.4370 |                          -0.0456 |

Inferencia ordinal:

- `ordinal_qwk_pairwise.csv`: nenhuma comparacao significativa apos Holm (`p_wilcoxon_holm` minimo = `0.9375`).
- `ordinal_qwk_baseline.csv`: nenhuma comparacao significativa apos Holm (`p_wilcoxon_holm` minimo = `0.3750`).

Leitura direta:

- Remover a classe `88` reduz o QWK absoluto em todos os classificadores nesta rodada.
- Mesmo no cenario ordinal, nao houve evidencia de superioridade robusta entre imputadores apos correcao multipla.

### Relatorio por classe do melhor modelo (MICE_XGBoost + XGBoost)

| Classe       | Precision        | Recall           | F1-Score         | Support |
| :----------- | :--------------- | :--------------- | :--------------- | ------: |
| 0            | 0.4816 +- 0.0080 | 0.7358 +- 0.0038 | 0.5821 +- 0.0055 |   19563 |
| 1            | 0.7473 +- 0.0029 | 0.6063 +- 0.0040 | 0.6694 +- 0.0035 |   71784 |
| 2            | 0.4316 +- 0.0042 | 0.5413 +- 0.0018 | 0.4803 +- 0.0032 |   32133 |
| 3            | 0.3719 +- 0.0050 | 0.5516 +- 0.0035 | 0.4442 +- 0.0046 |   26692 |
| 4            | 0.7213 +- 0.0055 | 0.6439 +- 0.0057 | 0.6804 +- 0.0056 |   60747 |
| 88           | 0.9459 +- 0.0012 | 0.8450 +- 0.0084 | 0.8926 +- 0.0052 |  106353 |
| macro avg    | 0.6166 +- 0.0043 | 0.6540 +- 0.0037 | 0.6248 +- 0.0045 |  317272 |
| weighted avg | 0.7289 +- 0.0033 | 0.6903 +- 0.0051 | 0.7028 +- 0.0046 |  317272 |

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
- `imputation_effect/`
  - `pairwise_global_f1_weighted.csv`
  - `pairwise_by_classifier_f1_weighted.csv`
  - `baseline_global_f1_weighted.csv`
  - `baseline_by_classifier_f1_weighted.csv`
  - `manifest_f1_weighted.json`
- `ordinal_sensitivity/`
  - `ordinal_metrics_by_fold.csv`
  - `ordinal_metrics_summary.csv`
  - `ordinal_qwk_pairwise.csv`
  - `ordinal_qwk_baseline.csv`
  - `manifest_ordinal.json`

Resultados brutos (`results/raw/`):

- `all_results.csv`
- `all_results_detailed.json`
- `checkpoint_classification_<mode>.json` (ex.: `checkpoint_classification_hybrid.json`)
- Para modos nao-default: `all_results_<mode>.csv`, `all_results_detailed_<mode>.json`
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

- Nesta rodada (`hybrid`, `n efetivo=1,586,358`), nao houve falhas de fold, mas metodos com imputacao pesada (principalmente `kNN` e `MissForest`) elevaram bastante o tempo total.
- Mesmo com `n_outer_folds=5`, os testes post-hoc com correcao multipla permaneceram sem significancia entre pares.
- A qualidade final depende diretamente da qualidade do dicionario de codigos e da consistencia da base de origem.
- A analise de efeito foi executada com `bootstrap=5000`; para estimativas ainda mais estaveis, pode-se aumentar esse valor.
- A analise `ordinal_sensitivity` tambem foi executada com `bootstrap=5000` e nao mostrou diferencas robustas entre imputadores apos Holm.

## Discussao e implicacoes

- Pergunta central: "imputacao melhora a classificacao?". Nesta execucao, a resposta e "nao de forma clara": as diferencas de F1 foram pequenas e sem significancia apos correcao multipla.
- Do ponto de vista pratico-operacional, o melhor F1 desta rodada (`MICE_XGBoost + XGBoost`) veio com custo computacional bem maior que variantes simples (`Media/Mediana + XGBoost`) e sem evidencia estatistica robusta de superioridade.
- No recorte ordinal (QWK), o padrao se manteve: variacoes pequenas entre imputadores e ausencia de significancia apos ajuste multiplo.
- Para consolidar uma conclusao metodologica mais forte, recomenda-se reportar em conjunto o efeito preditivo global e a estabilidade ordinal.

## Proximos passos sugeridos

- Aumentar `n_outer_folds` (ex.: 7) ou usar repeticoes de CV para fortalecer inferencia estatistica.
- Aumentar `bootstrap` (ex.: `10000`) nas analises inferenciais para reduzir incerteza dos ICs.
- Testar calibracao de probabilidades para AUC multiclasses.
- Executar analises estratificadas por subgrupos clinicos quando aplicavel.
