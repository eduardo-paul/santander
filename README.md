# Satisfação dos clientes do Santander.
Projeto da Formação Cientista de Dados da [Data Science Academy](https://www.datascienceacademy.com.br/), inspirado por uma competição do [Kaggle](https://www.kaggle.com/c/santander-customer-satisfaction/).

## Descrição do problema e objetivos.
Este conjunto apresenta dados de 76020 clientes, para os quais são retratadas 369 variáveis. Como esses dados são relativos a clientes reais, toda a informação foi anonimizada, de forma que os nomes das variáveis, assim como seus valores, pouco ajudam a entender o problema.

Neste conjunto de dados, a variável "ID" identifica cada cliente, e a variável "TARGET" indica a sua satisfação, onde 0 é "satisfeito" e 1 é "insatisfeito." O objetivo aqui é detectar corretamente a insatisfação do cliente, que definimos então como a classe positiva.

## Considerações.
Explorando os dados, é possível perceber que o problema é altamente desbalanceado, tendo apenas cerca de 4% dos clientes classificados como insatisfeitos. Esse desbalancemento traz dificuldades para a obtenção de um modelo de classificação. Por exemplo, um modelo que preveja que nunca um cliente estará insatisfeito terá 96% de acurácia. Apesar de este valor num primeiro momento sugerir um modelo de sucesso, o fato de ele nunca ser capaz de detectar a insatisfação de um cliente torna-o inútil para o nosso propósito.

Uma medida mais interessante seria a sensibilidade, que mede a taxa de acerto entre os casos positivos (quantos dos clientes insatisfeitos foram corretamente classificados). Em outras palavras, é uma medida de quanto o modelo é _sensível_, ou o quanto ele é capaz de detectar, a insatisfação.

De maneira complementar à sensibilidade, também podemos definir a especificidade. Esta métrica resume a taxa de acerto do modelo nos casos negativos (clientes satisfeitos). O modelo é bastante _específico_ quando ele detecta apenas os casos que são verdadeiramente positivos. Em outras palavras, ele é capaz de "enxergar" _especificamente_ a condição procurada, não se deixando enganar por distrações.

Para levar em consideração as duas métricas descritas, podemos utilizar a média geométrica.