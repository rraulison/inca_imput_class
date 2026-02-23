# Pipeline de Imputacao e Classificacao para Estadiamento Oncologico (SisRHC/INCA)

Este README foi atualizado para refletir **apenas** o escopo do arquivo:

- `results/raw/all_results_detailed_hybrid.json`
- ultima atualizacao do arquivo: `2026-02-23 12:27:11 -0300`

## Escopo deste documento

Incluido:

- resultados fold-a-fold do modo `hybrid`
- comparacao entre combinacoes `imputer + classifier`
- metricas de classificacao e tempo presentes no JSON
- relatorio por classe e matriz de confusao agregada do melhor modelo

Excluido (fora do escopo deste JSON):

- validacao temporal
- TabICL
- analise de efeito da imputacao (`imputation_effect_stats`)
- sensibilidade ordinal (`ordinal_sensitivity`)
- testes estatisticos externos que dependem de outros artefatos

## Como reproduzir este artefato

Pre-requisito: executar `prepare` e `impute` antes de `classify`.

```bash
python main.py --step classify --runtime-mode hybrid
```

Saida principal desta etapa:

- `results/raw/all_results_detailed_hybrid.json`

## Estrutura do JSON

Cada item do array representa 1 fold de 1 combinacao e contem:

- identificacao: `fold`, `imputer`, `classifier`, `runtime_mode`
- metricas: `accuracy`, `recall_weighted`, `f1_weighted`, `auc_weighted`, `recall_macro`, `f1_macro`, `auc_macro`
- tempos: `time_imputation_fit`, `time_imputation_transform`, `time_tuning`, `time_prediction`, `time_total`
- tuning: `best_params`, `best_inner_score`
- diagnostico: `classification_report`, `confusion_matrix`

## Resumo da execucao (extraido do JSON)

- `runtime_mode`: `hybrid`
- linhas no JSON: `100`
- folds: `5` (`0..4`)
- combinacoes avaliadas: `20`
- imputadores: `8` (`Media`, `Mediana`, `kNN`, `MICE`, `MICE_XGBoost`, `MissForest`, `NoImpute`, `RawSemEncoding`)
- classificadores: `4` (`XGBoost`, `CatBoost`, `cuML_RF`, `cuML_SVM`)
- suporte medio por fold: `200000`
- total avaliado (5 folds): `1000000`

## Ranking das combinacoes (media dos 5 folds)

Ordenado por `f1_weighted` (maior para menor).

| imputer | classifier | accuracy mean+/-std | f1_weighted mean+/-std | auc_weighted mean+/-std | time_total mean (s) |
|---|---:|---:|---:|---:|---:|
| Media | XGBoost | 0.5630 +/- 0.0030 | 0.5687 +/- 0.0029 | 0.8608 +/- 0.0021 | 49.44 |
| Mediana | XGBoost | 0.5629 +/- 0.0028 | 0.5687 +/- 0.0028 | 0.8607 +/- 0.0021 | 49.55 |
| kNN | XGBoost | 0.5626 +/- 0.0031 | 0.5684 +/- 0.0030 | 0.8603 +/- 0.0021 | 404.80 |
| MICE_XGBoost | XGBoost | 0.5629 +/- 0.0114 | 0.5684 +/- 0.0105 | 0.8604 +/- 0.0075 | 283.95 |
| MICE | XGBoost | 0.5624 +/- 0.0030 | 0.5682 +/- 0.0029 | 0.8605 +/- 0.0020 | 111.74 |
| MissForest | XGBoost | 0.5622 +/- 0.0029 | 0.5679 +/- 0.0028 | 0.8605 +/- 0.0021 | 805.87 |
| NoImpute | XGBoost | 0.5612 +/- 0.0095 | 0.5671 +/- 0.0090 | 0.8598 +/- 0.0064 | 50.19 |
| RawSemEncoding | CatBoost | 0.5549 +/- 0.0044 | 0.5610 +/- 0.0040 | 0.8576 +/- 0.0031 | 111.30 |
| MICE | cuML_RF | 0.5523 +/- 0.0008 | 0.5455 +/- 0.0008 | 0.8433 +/- 0.0003 | 72.49 |
| kNN | cuML_RF | 0.5512 +/- 0.0006 | 0.5440 +/- 0.0005 | 0.8427 +/- 0.0003 | 363.04 |
| MICE_XGBoost | cuML_RF | 0.5510 +/- 0.0004 | 0.5437 +/- 0.0006 | 0.8425 +/- 0.0001 | 243.31 |
| MissForest | cuML_RF | 0.5511 +/- 0.0004 | 0.5435 +/- 0.0003 | 0.8425 +/- 0.0003 | 764.09 |
| Mediana | cuML_RF | 0.5503 +/- 0.0007 | 0.5418 +/- 0.0009 | 0.8418 +/- 0.0002 | 8.86 |
| Media | cuML_RF | 0.5503 +/- 0.0007 | 0.5418 +/- 0.0009 | 0.8418 +/- 0.0002 | 8.84 |
| MICE_XGBoost | cuML_SVM | 0.4219 +/- 0.0008 | 0.4042 +/- 0.0024 | 0.7290 +/- 0.0013 | 270.44 |
| Media | cuML_SVM | 0.4196 +/- 0.0011 | 0.4015 +/- 0.0004 | 0.7258 +/- 0.0007 | 35.58 |
| Mediana | cuML_SVM | 0.4196 +/- 0.0011 | 0.4015 +/- 0.0004 | 0.7258 +/- 0.0007 | 35.62 |
| MissForest | cuML_SVM | 0.4186 +/- 0.0015 | 0.4003 +/- 0.0016 | 0.7263 +/- 0.0009 | 791.86 |
| MICE | cuML_SVM | 0.4167 +/- 0.0017 | 0.3990 +/- 0.0014 | 0.7237 +/- 0.0011 | 99.74 |
| kNN | cuML_SVM | 0.4153 +/- 0.0007 | 0.3970 +/- 0.0010 | 0.7234 +/- 0.0009 | 392.13 |

## Media por classificador (agregando combinacoes disponiveis no JSON)

| classifier | accuracy mean+/-std | f1_weighted mean+/-std | auc_weighted mean+/-std | time_total mean (s) |
|---|---:|---:|---:|---:|
| XGBoost | 0.5624 +/- 0.0056 | 0.5682 +/- 0.0053 | 0.8604 +/- 0.0038 | 250.79 |
| CatBoost | 0.5549 +/- 0.0044 | 0.5610 +/- 0.0040 | 0.8576 +/- 0.0031 | 111.30 |
| cuML_RF | 0.5511 +/- 0.0009 | 0.5434 +/- 0.0015 | 0.8424 +/- 0.0006 | 243.44 |
| cuML_SVM | 0.4186 +/- 0.0025 | 0.4006 +/- 0.0026 | 0.7257 +/- 0.0021 | 270.90 |

## Melhor combinacao neste arquivo

Melhor `f1_weighted` medio:

- `Media + XGBoost`
- `f1_weighted`: `0.5687 +/- 0.0029`
- `accuracy`: `0.5630 +/- 0.0030`
- `auc_weighted`: `0.8608 +/- 0.0021`
- `time_total`: `49.44 s` (media por fold)

### Relatorio medio por classe (5 folds) - `Media + XGBoost`

| classe | precision mean+/-std | recall mean+/-std | f1 mean+/-std | suporte medio |
|---|---:|---:|---:|---:|
| 0 | 0.2813 +/- 0.0038 | 0.6845 +/- 0.0030 | 0.3987 +/- 0.0039 | 8448 |
| 1 | 0.6678 +/- 0.0042 | 0.5077 +/- 0.0026 | 0.5768 +/- 0.0030 | 40334 |
| 2 | 0.4357 +/- 0.0032 | 0.4442 +/- 0.0043 | 0.4399 +/- 0.0036 | 36625 |
| 3 | 0.3753 +/- 0.0023 | 0.3330 +/- 0.0047 | 0.3528 +/- 0.0033 | 32724 |
| 4 | 0.5226 +/- 0.0040 | 0.5690 +/- 0.0023 | 0.5448 +/- 0.0025 | 35965 |
| 5 | 0.8952 +/- 0.0038 | 0.8433 +/- 0.0054 | 0.8684 +/- 0.0044 | 45904 |
| macro avg | 0.5296 +/- 0.0027 | 0.5636 +/- 0.0025 | 0.5302 +/- 0.0029 | 200000 |
| weighted avg | 0.5872 +/- 0.0028 | 0.5630 +/- 0.0030 | 0.5687 +/- 0.0029 | 200000 |

### Matriz de confusao agregada (soma dos 5 folds) - `Media + XGBoost`

Linhas = classe real, colunas = classe predita.

```text
28911  5318  3603  2108  1230  1069
29411 102385 33982 15769 15731  4394
18279 18865 81341 36886 23225  4527
14497 12175 39039 54479 38366  5065
 7954  8384 24213 29352 102317 7607
 3756  6202  4516  6572 14929 193543
```

## Artefatos no escopo

- `results/raw/all_results_detailed_hybrid.json` (fonte unica deste README)
- `results/raw/all_results_hybrid.csv` (se gerado na mesma execucao)
- `results/raw/checkpoint_classification_hybrid.json` (progresso da etapa de classificacao)

## Observacao

As conclusoes acima refletem **somente** o que esta dentro de `all_results_detailed_hybrid.json`.
