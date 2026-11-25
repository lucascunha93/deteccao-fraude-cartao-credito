# Detecção de fraude em cartão de crédito

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucascunha93/deteccao-fraude-cartao-credito/blob/main/deteccao_fraude_cartao.ipynb)

Projeto do desafio da DIO: treinar modelos que detectam fraude em transações reais de cartão de crédito, comparar algoritmos pelas métricas que importam quando a fraude é rara e explicar o que levou uma transação a ser marcada como fraude.

Tudo está no notebook [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb), executado no Google Colab do início ao fim com as tabelas e os gráficos salvos.

**Resultado em uma linha:** o XGBoost ajustado, com limiar de decisão 0,256, encontrou **74 das 95 fraudes** do conjunto de teste (recall 0,779) com **precisão de 0,937**: foram só 5 alarmes falsos em 56.746 transações.

---

## 1. O problema e por que o desbalanceamento muda a avaliação

A base tem 284.807 transações de dois dias de cartões europeus. `Time` marca o momento (segundos desde a primeira transação), `Amount` o valor e `Class` a resposta (1 = fraude). As variáveis `V1` a `V28` vieram de um PCA aplicado para proteger a privacidade.

| Classe | Transações | Proporção |
|---|---:|---:|
| Normal (0) | 284.315 | 99,83% |
| Fraude (1) | 492 | 0,17% |

Com essa proporção, **a acurácia engana**. Um classificador que responde "normal" para tudo tem **99,83% de acurácia** e não encontra nenhuma fraude. O notebook mostra isso logo no início com um `DummyClassifier`.

Por isso a avaliação olha para a **classe de fraude**:

- **Recall**: das fraudes reais, quantas o modelo encontrou. É a métrica principal, porque uma fraude perdida é dinheiro perdido.
- **Precisão**: das transações marcadas como fraude, quantas eram fraude. Precisão baixa significa uma fila cheia de alarmes falsos e clientes com o cartão bloqueado à toa.
- **F1**: média harmônica das duas.
- **PR-AUC** (average precision): área sob a curva de precisão e recall. Resume a qualidade do modelo em todos os limiares. Ao contrário da ROC-AUC, ela não fica inflada pelas milhares de transações normais fáceis.

## 2. Preparação dos dados

1. **Remoção de 1.081 linhas duplicadas.** Assim a mesma transação não cai no treino e no teste ao mesmo tempo. Sobram 283.726 linhas e 473 fraudes.
2. **`Amount_log = log1p(Amount)`.** O valor é muito assimétrico: mediana de 22 e máximo de 25.691.
3. **Divisão treino/teste 80/20 com `stratify`.** As duas partes ficam com 0,17% de fraude (378 no treino, 95 no teste).
4. **`StandardScaler` ajustado só no treino.** Nas etapas seguintes ele fica dentro de um `Pipeline`, então também é reajustado dentro de cada dobra da validação cruzada.
5. **Variáveis de comportamento ao longo do tempo** (para um teste da seção 6): hora relativa, segundos desde a transação anterior, número de transações no último minuto e valor comparado com a média das 10 transações anteriores. Todas olham só para o passado de cada transação.

Os modelos de árvore recebem os dados sem padronização, porque árvores não dependem de escala e assim o SHAP mostra os valores originais. A regressão logística recebe os dados padronizados.

## 3. Comparação entre os modelos

### 3.1 Primeira rodada (limiar padrão 0,5, conjunto de teste)

| Modelo | Precisão | Recall | F1 | PR-AUC | Fraudes pegas | Alarmes falsos |
|---|---:|---:|---:|---:|---:|---:|
| Sempre "normal" | 0,000 | 0,000 | 0,000 | 0,002 | 0/95 | 0 |
| Regressão logística | 0,849 | 0,590 | 0,696 | 0,700 | 56/95 | 10 |
| Regressão logística (`class_weight='balanced'`) | 0,055 | 0,874 | 0,104 | 0,677 | 83/95 | 1.422 |
| Random Forest (`class_weight='balanced'`) | 0,986 | 0,726 | 0,836 | 0,815 | 69/95 | 1 |
| XGBoost | 0,973 | 0,758 | 0,852 | **0,835** | 72/95 | 2 |
| XGBoost (`scale_pos_weight`) | 0,904 | 0,790 | 0,843 | 0,820 | 75/95 | 8 |

- A regressão logística com peso `balanced` tem o maior recall, mas paga com **1.422 alarmes falsos**: só 5,5% dos alertas eram fraude.
- Os modelos de árvore (Random Forest e XGBoost) separam muito melhor as classes. O XGBoost tem a melhor PR-AUC.
- Na **curva ROC** todos os modelos parecem ótimos (AUC entre 0,94 e 0,97). A **curva de precisão e recall** é a que mostra a diferença real entre eles.

### 3.2 Comparação final (cada modelo no seu limiar, conjunto de teste)

O limiar de cada modelo foi escolhido **sem olhar o teste** (ver seção 4).

| Modelo | Limiar | Precisão | Recall | F1 | PR-AUC | Fraudes pegas | Alarmes falsos |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,050 | 0,856 | **0,811** | 0,832 | **0,835** | 77/95 | 13 |
| Random Forest no Pipeline (`balanced`) | 0,190 | 0,926 | 0,789 | **0,852** | 0,815 | 75/95 | 6 |
| **XGBoost ajustado (GridSearchCV)** | **0,256** | **0,937** | 0,779 | 0,851 | 0,829 | 74/95 | **5** |
| XGBoost ajustado + variáveis de tempo | 0,119 | 0,841 | 0,779 | 0,809 | 0,818 | 74/95 | 14 |
| Extra Trees (`balanced`) | 0,394 | 0,901 | 0,768 | 0,830 | 0,811 | 73/95 | 8 |
| Regressão logística no Pipeline | 0,045 | 0,794 | 0,768 | 0,781 | 0,700 | 73/95 | 19 |
| HistGradientBoosting | 0,725 | 0,624 | 0,716 | 0,667 | 0,622 | 68/95 | 41 |

O teste tem só 95 fraudes, então cada fraude vale cerca de 1 ponto de recall. Diferenças de 2 ou 3 fraudes entre os três primeiros modelos estão dentro do ruído.

### 3.3 Undersampling × oversampling

A amostragem foi aplicada só no treino; o teste manteve a proporção real.

| Modelo | Estratégia | Linhas de treino | Precisão | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Regressão logística | sem amostragem | 226.980 | 0,849 | 0,590 | 0,696 | 0,700 |
| Regressão logística | undersampling | 756 | 0,049 | 0,874 | 0,093 | 0,422 |
| Regressão logística | oversampling | 453.204 | 0,055 | 0,874 | 0,104 | 0,674 |
| XGBoost | sem amostragem | 226.980 | 0,973 | 0,758 | 0,852 | 0,835 |
| XGBoost | undersampling | 756 | 0,052 | 0,874 | 0,099 | 0,691 |
| XGBoost | oversampling | 453.204 | 0,892 | 0,779 | 0,832 | 0,820 |

As duas técnicas sobem o recall, mas o undersampling joga fora 99,7% das transações normais, e a precisão despenca para cerca de 5%. O oversampling preserva melhor a precisão no XGBoost, mas não supera o modelo treinado com todos os dados e com limiar ajustado. Seguindo a regra de testar antes de escolher, fiquei **sem amostragem e com ajuste de limiar**.

### 3.4 GridSearchCV

- **Regressão logística** (5 valores de `C` × 4 pesos para a fraude, validação cruzada de 5 dobras, escolha pelo recall): a busca escolheu `class_weight='balanced'` com recall de 0,907 na validação cruzada, mas a precisão foi de 0,059. Escolher **só** pelo recall leva sempre ao modelo que marca tudo como suspeito. A melhor combinação por F1 foi `C=0.01` com peso 10 para a fraude: recall 0,820 e precisão 0,764.
- **XGBoost** (profundidade × taxa de aprendizado × `scale_pos_weight`, 3 dobras, escolha pela PR-AUC): o melhor foi `max_depth=5`, `learning_rate=0.05` e `scale_pos_weight=25`, com PR-AUC de 0,853 na validação cruzada.

## 4. Limiar de decisão escolhido

O limiar padrão de 0,5 não é sagrado. Baixar o limiar troca precisão por recall. Na regressão logística, passar de 0,5 para 0,3 levou o recall de 0,590 para 0,642 (de 56 para 61 fraudes) com só 3 alarmes falsos a mais.

Em vez de fixar 0,3 para todos os modelos, escolhi o limiar de cada modelo assim:

1. Gerei previsões **fora da dobra** no treino com `cross_val_predict` (3 dobras), sem tocar no teste.
2. Escolhi o limiar que maximiza o **F2**, que dá ao recall o dobro do peso da precisão. Ele segue a prioridade de pegar fraude sem deixar a precisão ir a zero, como acontece quando se escolhe só pelo recall.
3. Apliquei esse limiar ao teste.

**Limiar escolhido para o modelo final (XGBoost ajustado): 0,256.** No teste, esse limiar pegou 74 de 95 fraudes com 5 alarmes falsos.

Se o negócio aceitar mais alarmes falsos em troca de recall, o XGBoost padrão no limiar 0,05 pega 77 de 95 fraudes com 13 alarmes falsos.

## 5. O que o SHAP mostrou

Usei o `TreeExplainer` no XGBoost ajustado, numa amostra do teste com todas as 95 fraudes e 3.000 transações normais.

- **Quatro variáveis concentram a decisão:** V14 (impacto médio absoluto de 1,63), V4 (1,44), V12 (0,81) e V10 (0,61). Depois vêm V3, V11 e V8. A importância nativa das árvores aponta o mesmo grupo.
- **Direção:** valores **muito negativos** de V14, V12 e V10 e valores **altos** de V4 empurram a transação para fraude. Nos gráficos de dependência, a contribuição para fraude cresce rapidamente conforme V14 fica mais negativo.
- **O valor da compra pesa pouco** (`Amount_log` tem impacto médio de 0,24, na 13ª posição). O modelo não marca uma transação como fraude só porque ela é cara.
- **Uma fraude detectada** (probabilidade ≈ 1,0): V14 = −13,7 sozinho somou +5,05 log-odds. Em seguida vieram V10 (+1,58), V17 (+1,47), V4 (+1,43) e V12 (+1,36).
- **Uma fraude que passou** (probabilidade ≈ 0): V4 = −1,3 empurrou fortemente para "normal" (−2,76), e as demais variáveis estavam em faixas típicas de transações legítimas. Fraudes assim são indistinguíveis das normais com essas variáveis. É o limite do que o modelo consegue ver.

Como V1–V28 vêm de PCA, não dá para traduzir "V14 negativo" em uma regra de negócio. O SHAP mostra *quais* sinais o modelo usa e *em que direção*, não o significado deles.

## 6. O que mudei em relação ao que a Expert fez

| O que a Expert fez | O que eu fiz | Por quê |
|---|---|---|
| — | Removi 1.081 linhas duplicadas antes da divisão | Evita a mesma transação no treino e no teste |
| Baixou o limiar de 0,5 para 0,3 | Escolhi o limiar de cada modelo pelo F2 em previsões fora da dobra | Escolher o limiar olhando o teste deixaria as métricas otimistas; 0,3 não é o melhor para todos os modelos |
| GridSearchCV escolhendo pelo recall | Mantive o critério de recall na regressão logística e registrei precisão, F1 e PR-AUC de cada combinação | Mostra que escolher só pelo recall leva a precisão de 6% |
| Random Forest fora do Pipeline | Random Forest dentro do Pipeline com ajuste de limiar | Sugestão da própria Expert; o RF passou de 69 para 75 fraudes pegas com 6 alarmes falsos |
| Grade de hiperparâmetros da aula | Grade maior na regressão logística (20 combinações) e uma grade para o XGBoost (18 combinações) | Ideia de evolução do desafio |
| — | Undersampling × oversampling comparados no mesmo teste | Ideia de evolução do desafio |
| — | Variáveis de comportamento ao longo do tempo | Testadas e descartadas: pioraram a PR-AUC de 0,829 para 0,818 |
| — | HistGradientBoosting e Extra Trees | Mais modelos comparados pelo recall |
| — | PR-AUC em todas as tabelas | Resume o modelo em todos os limiares e não fica inflada pela classe majoritária |

## Como rodar

Clique no botão **Abrir no Colab** no topo desta página e use **Ambiente de execução → Executar tudo**. A base é lida direto do link do desafio:

```python
pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
```

O CSV **não** está no repositório (passa do limite de 100 MB do GitHub) e está listado no `.gitignore`. Na CPU gratuita do Colab, a execução completa leva cerca de 45 minutos. A maior parte desse tempo vai para a validação cruzada do Random Forest e para o GridSearchCV do XGBoost.

**Bibliotecas:** pandas, numpy, scikit-learn, xgboost, shap, matplotlib e seaborn (todas já vêm no Colab).

## Arquivos

- [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb): notebook completo, com as saídas salvas
- [`.gitignore`](.gitignore): mantém o CSV fora do repositório
- `README.md`: este arquivo
