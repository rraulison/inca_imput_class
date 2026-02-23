# Protocolo Cientifico: "Imputacao melhora a classificacao downstream?"

## 1) Objetivo e pergunta

Objetivo do experimento: responder, com inferencia estatistica e validade externa, se tecnicas de imputacao melhoram o desempenho de classificacao downstream em relacao aos baselines sem imputacao (quando suportado) e imputacao simples.

Pergunta primaria:

- Em media pareada por split, algum metodo de imputacao aumenta `f1_weighted` acima de uma margem pratica predefinida?

Hipotese primaria:

- `H0`: `delta_f1_weighted <= 0` (ou nao superior a margem pratica).
- `H1`: `delta_f1_weighted > 0` e ganho pratico relevante.

Distincao obrigatoria para o alvo ordinal:

- Classes `0-4`: tratadas como escala ordinal valida para metricas ordinais.
- Classe `88` ("tumor nao estadiavel"): tratada como classe especial nao ordinal.

Configuracao confirmatoria padrao:

- metrica primaria: `f1_weighted`
- metrica secundaria nominal: `f1_macro` (reportada no artigo para capturar efeito em classes minoritarias)
- metrica ordinal secundaria (somente `0-4`): `qwk_0_4`
- metricas da classe especial `88`: `f1_88`, `recall_88` e `f1_estadiavel_bin` (`0-4` vs `88`)
- alpha: `0.05`
- margem pratica primaria (equivalence/non-inferiority): `0.01`
- margem pratica de sensibilidade: `0.005` (reportada como analise complementar)
- desenho: `Repeated Stratified Nested CV` (5 outer folds x 5 repeticoes; inner 5 folds)
- validacao externa: holdout temporal final
- teste pareado primario: permutacao pareada (n_perms=10000)
- teste pareado secundario: Wilcoxon signed-rank

Justificativa das escolhas:

- Margem de `0.01`: margens de 0.005 em F1 podem ser indistinguiveis de ruido em classificacao multiclasse. A margem de 0.01 e mais conservadora e alinhada com a literatura. A margem de 0.005 e reportada como analise de sensibilidade.
- 5 repeticoes (em vez de 3): geram 25 observacoes pareadas, aumentando o poder estatistico do teste de permutacao e do Wilcoxon para detectar efeitos de magnitude 0.01.
- Permutacao pareada como teste primario: mais robusto que Wilcoxon para amostras pequenas-moderadas (n=25) e nao assume simetria da distribuicao dos deltas.

## 2) Definition of Done (DoD)

O protocolo so e considerado concluido quando todos os itens abaixo forem verdadeiros:

- [ ] Dados, filtros, classes e periodo congelados com hash/versionamento registrados.
- [ ] Splits repetidos gerados e persistidos com seeds fixas e reprodutiveis.
- [ ] Preprocessamento, encoding, imputacao e classificador executados dentro de pipeline unico no loop interno (sem leakage).
- [ ] Todos os metodos comparados sob orcamento de tuning equivalente (mesmo `n_iter` ou grid exaustivo para espacos pequenos).
- [ ] Baseline `NoImpute` confirmado como passando missings reais (`NaN`) a classifiers que suportam (XGBoost, CatBoost).
- [ ] Resultados por `(repeat, outer_fold, classifier, method)` salvos em artefato tabular unico.
- [ ] Analise pareada com IC bootstrap, teste de permutacao pareada, teste Wilcoxon, ajuste Holm e TOST executada.
- [ ] Analises separadas `with_88` e `without_88` executadas e reportadas.
- [ ] `QWK` calculado apenas no subconjunto ordinal (`0-4`) e metricas da classe `88` reportadas separadamente.
- [ ] Holdout temporal executado apenas para confirmacao final.
- [ ] Relatorio final com criterio de decisao explicito: "melhora / nao melhora / equivalente".

## 3) Checklist operacional (execucao)

Sequencia de comandos recomendada (estado atual do repositorio):

```bash
python main.py --step prepare
python main.py --step impute
python main.py --step classify --runtime-mode default
python src/run_imputation_effect_stats.py --metric f1_weighted --equivalence-margin 0.01 --baseline NoImpute
python src/run_ordinal_sensitivity.py --baseline NoImpute --bootstrap-iters 5000
python src/run_temporal_sensitivity.py --runtime-mode default
```

Sequencia alvo apos adequacao (modo confirmatorio):

```bash
python main.py --step protocol --protocol-mode confirmatory
```

Dry-run recomendado antes do experimento completo:

```bash
python main.py --step protocol --protocol-mode confirmatory --dry-run \
    --n-sample 5000 --repeats 1 --protocol-imputer Media,NoImpute --protocol-classifier XGBoost
```

### Fase A - Pre-registro e congelamento

- [ ] Criar manifesto do estudo em `results/tables/protocol/manifest_protocol.json`.
- [ ] Registrar: dataset hash, codigo hash (commit), config hash, seed schedule.
- [ ] Fixar comparadores e metrica primaria antes da rodada confirmatoria.
- [ ] Fixar politica para classe `88`: `qwk_0_4` no subconjunto ordinal e metricas separadas para `88`.
- [ ] Definir seed schedule explicitamente: `seeds: [42, 123, 456, 789, 1024]`.
- [ ] Persistir indices dos splits como artefato para reprodutibilidade.
- [ ] Estimar e registrar tempo total esperado do experimento.

Artefatos minimos esperados:

- `results/tables/protocol/manifest_protocol.json`
- `results/raw/pip_freeze.txt`
- `results/tables/metadata.json`
- `results/raw/protocol_splits.pkl` (indices dos splits persistidos)

### Fase B - Implementacao experimental confirmatoria

NOTA CRITICA — LEAKAGE DE IMPUTACAO:
O pipeline atual em `run_classification.py` consome dados ja imputados da pasta `data/imputed/`.
Isso significa que a imputacao e feita antes do split outer, usando dados de teste no fit do imputer.
O caminho confirmatorio DEVE:

1. Receber dados crus (com missings) no loop outer.
2. Fit imputer apenas em `X_train_outer`.
3. Transform tanto `X_train_outer` quanto `X_test_outer`.
4. So entao executar tuning/treino/avaliacao no loop interno.

- [ ] Implementar modo `confirmatory` com repeated nested CV.
- [ ] Implementar pipeline sem leakage: imputacao dentro do loop outer (fit em train, transform em test).
- [ ] Executar tuning interno com pipeline unico (`encoder -> imputer -> scaler -> model`).
- [ ] Garantir pareamento estrito por split: todas as combinacoes (imputer, classifier) usam exatamente os mesmos splits em cada repeticao.
- [ ] Padronizar orcamento de tuning: `n_iter=30` para todos os classificadores, ou grid exaustivo para espacos com <30 combinacoes (ex: cuML_SVM com ~10 combinacoes).
- [ ] Persistir resultados fold-a-fold com colunas de pareamento.
- [ ] Implementar checkpointing por combinacao `(repeat, outer_fold, classifier, imputer)` para retomada.

Artefato minimo esperado:

- `results/raw/protocol_results.csv`

Schema minimo sugerido para `protocol_results.csv`:

- `repeat`
- `outer_fold`
- `classifier`
- `imputer`
- `method_id`
- `accuracy`
- `f1_weighted`
- `f1_macro`
- `auc_weighted`
- `qwk_0_4`
- `f1_88`
- `recall_88`
- `f1_estadiavel_bin`
- `time_fit`
- `time_predict`
- `time_total`
- `best_params`
- `seed`
- `runtime_mode`
- `n_train`
- `n_test`

### Fase C - Inferencia estatistica confirmatoria

- [ ] Calcular `delta` pareado por unidade `(repeat, outer_fold, classifier)`.
- [ ] Rodar IC bootstrap de `delta_mean` (seeds deterministicas por comparacao).
- [ ] Rodar teste de permutacao pareada (primario, n_perms=10000).
- [ ] Rodar teste Wilcoxon signed-rank (secundario).
- [ ] Ajustar p-valores para multiplas comparacoes (Holm).
- [ ] Rodar TOST e nao-inferioridade com margem primaria (0.01) e sensibilidade (0.005).
- [ ] Rodar analise ordinal secundaria (`qwk_0_4`) no recorte `without_88`.
- [ ] Rodar analise de seguranca da classe especial (`f1_88`, `recall_88`, `f1_estadiavel_bin`) no recorte `with_88`.
- [ ] Reportar `f1_macro` como metrica secundaria no corpo do artigo.

Artefatos minimos esperados:

- `results/tables/protocol/baseline_global_f1_weighted.csv`
- `results/tables/protocol/baseline_by_classifier_f1_weighted.csv`
- `results/tables/protocol/pairwise_global_f1_weighted.csv`
- `results/tables/protocol/pairwise_by_classifier_f1_weighted.csv`
- `results/tables/protocol/baseline_global_qwk_0_4.csv`
- `results/tables/protocol/pairwise_global_qwk_0_4.csv`
- `results/tables/protocol/class88_guardrails.csv`

### Fase D - Validacao externa temporal

- [ ] Selecionar metodo final apenas com base na CV.
- [ ] Treinar no periodo antigo e testar no periodo recente (holdout temporal).
- [ ] Nao reutilizar holdout temporal para tuning.
- [ ] Aplicar pipeline sem leakage tambem no holdout temporal (fit imputer no treino, transform no teste).

Artefatos minimos esperados:

- `results/raw/temporal_sensitivity_results.csv`
- `results/tables/temporal_sensitivity/summary_temporal.csv`

### Fase E - Decisao final e relatorio

- [ ] Declarar resultado final com regra explicita (ganho pratico, significancia corrigida, consistencia e confirmacao temporal).
- [ ] Declarar explicitamente a conclusao para `without_88` (ordinal) e `with_88` (classe especial).
- [ ] Publicar relatorio curto em `results/tables/protocol/conclusion.md`.

## 4) Mudancas de codigo por arquivo (plano de implementacao)

Prioridade P0 (obrigatorio para protocolo confirmatorio):

0. `src/run_protocol.py` [NOVO] [SPRINT 0]

- Orquestrador unico do protocolo confirmatorio.
- Recebe dados crus, executa imputacao dentro do loop outer (sem leakage).
- Interface CLI para rodar fim-a-fim e gerar manifesto final.
- Pipeline confirmatório: dados crus -> split outer -> fit imputer em train -> transform train+test -> encoding -> scaling -> tuning inner -> avaliacao outer.

1. `config/config.yaml`

- Adicionar secao `protocol`:
  - `mode: exploratory|confirmatory`
  - `primary_metric: f1_weighted`
  - `secondary_metrics: [f1_macro]`
  - `alpha: 0.05`
  - `equivalence_margin: 0.01`
  - `sensitivity_margin: 0.005`
  - `outer_folds: 5`
  - `inner_folds: 5`
  - `repeats: 5`
  - `seed_schedule: [42, 123, 456, 789, 1024]`
  - `tuning_budget: 30`
  - `permutation_n_perms: 10000`

2. `main.py`

- Adicionar `--step protocol` e flags para `--protocol-mode`, `--repeats`, `--alpha`, `--equivalence-margin`, `--dry-run`.
- Encadear execucao confirmatoria: classificacao confirmatoria -> analise estatistica -> temporal holdout.

3. `src/run_classification.py`

- Criar caminho de execucao confirmatorio com repeated nested CV.
- Mover preprocessamento/imputacao para pipeline treinado no loop interno.
- Salvar resultado com chaves de pareamento: `repeat`, `outer_fold`, `classifier`, `imputer`, `seed`, `runtime_mode`, `n_train`, `n_test`.

4. `src/run_imputation_effect_stats.py`

- Aceitar e usar `repeat` + `outer_fold` como unidade pareada.
- Produzir tabelas separadas para modo confirmatorio em `results/tables/protocol/`.
- Incluir analises separadas por recorte: `with_88` e `without_88`.
- Incluir teste de permutacao pareada como teste primario.
- Rodar TOST com ambas as margens (0.01 e 0.005).

Prioridade P1 (consistencia e rastreabilidade):

5. `src/run_temporal_sensitivity.py`

- Consumir metodo selecionado na fase confirmatoria.
- Garantir separacao estrita entre selecao (CV) e confirmacao (temporal).
- Aplicar pipeline sem leakage (fit imputer no treino temporal).

6. `src/run_analysis.py`

- Gerar sumarios especificos do protocolo (tabela principal confirmatoria, trade-off desempenho x tempo, criterios de decisao).
- Gerar secao dedicada para metricas ordinais (`qwk_0_4`) e guardrails da classe `88`.

7. `src/stats_utils.py`

- Incluir teste de permutacao pareada (`permutation_test_paired`).
- Garantir funcoes de IC bootstrap com seed deterministica por comparacao.

8. `src/run_ordinal_sensitivity.py`

- Promover para etapa confirmatoria secundaria com saidas em `results/tables/protocol/`.
- Separar claramente os relatorios `with_88` e `without_88`.

Prioridade P2 (qualidade e automacao):

9. `tests/`

- `tests/test_protocol_splits.py`: reprodutibilidade dos splits e pareamento.
- `tests/test_no_leakage_pipeline.py`: ausencia de leakage entre inner/outer.
- `tests/test_protocol_stats.py`: consistencia de IC, Holm, TOST, permutacao.
- `tests/test_temporal_holdout_protocol.py`: separacao treino/teste temporal.
- `tests/test_ordinal_with_without_88.py`: validacao da separacao `0-4` vs `88`.

## 5) Estimativa de custo computacional

Com ~100k amostras, 7 imputadores x 4 classificadores x 5 folds x 5 repeticoes x 5 inner folds:

- ~3500 treinos de classificadores (inner CV x tuning)
- ~700 imputacoes (imputadores caros: MICE, MissForest, kNN)
- Estimativa total: 48-120h dependendo do hardware e imputadores

Mitigacoes:

- Checkpointing por combinacao `(repeat, outer_fold, classifier, imputer)` para retomada.
- Dry-run obrigatorio em amostra pequena antes do experimento completo.
- Considerar early stopping ou `max_iter` reduzido para MICE/MissForest se tempo > budget.

## 6) Cronograma recomendado

Sprint 0 (pre-requisito critico):

- Pipeline sem leakage: `run_protocol.py` com imputacao dentro do loop outer
- Dry-run para validacao

Sprint 1 (P0):

- configuracao de protocolo em `config.yaml`
- repeated nested CV confirmatorio (5 rep x 5 folds)
- orcamento de tuning padronizado
- analise pareada com teste de permutacao + ajuste multiplo

Sprint 2 (P1):

- temporal holdout confirmatorio com pipeline sem leakage
- relatorios de decisao e rastreabilidade

Sprint 3 (P2):

- suite de testes automatizados
- robustez, manutencao e documentacao final

## 7) Criterio de resposta a pergunta cientifica

Responder "imputacao melhora a classificacao downstream" somente se:

- `delta_mean` da metrica primaria for positivo e acima da margem pratica definida (0.01)
- houver evidencia estatistica apos correcao para multiplas comparacoes (Holm-adjusted p < 0.05)
- o teste de permutacao pareada confirmar o resultado
- o efeito for consistente entre repeticoes/folds
- o ganho se mantiver no holdout temporal
- no recorte `without_88`, houver melhora ou nao piora relevante em `qwk_0_4`
- no recorte `with_88`, nao houver degradacao relevante em `f1_88` e `recall_88`

Caso contrario, responder como:

- "nao melhora de forma robusta" ou
- "equivalente ao baseline com maior custo computacional" (se TOST confirmar equivalencia na margem de 0.01)

Analise de sensibilidade adicional:

- Repetir conclusao com margem de 0.005 e reportar como analise complementar.
- Reportar `f1_macro` para verificar se efeito e mascarado por classes majoritarias em `f1_weighted`.
