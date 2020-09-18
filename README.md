# Satisfação dos clientes do Santander.
Projeto da Formação Cientista de Dados da [Data Science Academy](https://www.datascienceacademy.com.br/), inspirado por uma competição do [Kaggle](https://www.kaggle.com/c/santander-customer-satisfaction/).

## Descrição do problema e objetivos.
Nesta competição, o banco Santander propôs o desafio de classificar os seus clientes entre satisfeitos e insatisfeitos. A capacidade de antever a insatisfação de um cliente é uma competência fundamental para qualquer negócio. De posse dessa informação, o banco pode ser proativo no sentido de melhorar a experiência do cliente, antes que ele decida abandonar a empresa.

Este conjunto apresenta dados de 76020 clientes, para os quais são retratadas 369 variáveis. Como esses dados são relativos a clientes reais, toda a informação foi anonimizada, de forma que os nomes das variáveis, assim como seus valores, pouco ajudam a entender o problema.

Neste conjunto de dados, a variável "ID" identifica cada cliente, e a variável "TARGET" indica a sua satisfação, onde 0 é "satisfeito" e 1 é "insatisfeito." O objetivo aqui é detectar corretamente a insatisfação do cliente, que definimos então como a classe positiva.

A informação de que um cliente está potencialmente insatisfeito não é suficiente por si só. Para a empresa poder tomar uma atitude adequada, é necessário que se compreenda qual a razão para a insatisfação. Não adiantaria o banco oferecer uma taxa de juros menor para um cliente que está insatisfeito apenas porque o tempo de espera na fila é muito alto.

Por essa razão, neste problema é necessário que obtenhamos um modelo de classificação que seja interpretável. Tal modelo pode fornecer insights sobre o sentimento do cliente e motivar uma melhor decisão de negócios.

## Considerações.
Explorando os dados, é possível perceber que o problema é altamente desbalanceado, tendo apenas cerca de 4% dos clientes classificados como insatisfeitos. Esse desbalancemento traz dificuldades para a obtenção de um modelo de classificação. Por exemplo, um modelo que preveja que nunca um cliente estará insatisfeito terá 96% de acurácia. Apesar de este valor num primeiro momento sugerir um modelo de sucesso, o fato de ele nunca ser capaz de detectar a insatisfação de um cliente torna-o inútil para o nosso propósito.

Uma medida mais interessante seria a sensibilidade, que mede a taxa de acerto entre os casos positivos (quantos dos clientes insatisfeitos foram corretamente classificados). Em outras palavras, é uma medida de quanto o modelo é _sensível_, ou o quanto ele é capaz de detectar, a insatisfação.

De maneira complementar à sensibilidade, também podemos definir a especificidade. Esta métrica resume a taxa de acerto do modelo nos casos negativos (clientes satisfeitos). O modelo é bastante _específico_ quando ele detecta apenas os casos que são verdadeiramente positivos. Em outras palavras, ele é capaz de "enxergar" _especificamente_ a condição procurada, não se deixando enganar por distrações.

Definir qual dessas duas métricas é a mais relevante é uma questão difícil e que precisa ser cuidadosamente discutida com os _stakeholders_. Caso nos foquemos em maximizar a sensibilidade, podemos acabar tendo um número muito grande de falsos positivos. Se o custo de um falso positivo for muito alto do ponto de vista do negócio, essa estratégia pode não valer a pena. Por outro lado, favorecer a especificidade reduzirá o número de falsos positivos, mas também diminuirá o número de clientes insatisfeitos que serão detectados.

Nesta análise, nós utilizamos uma árvore de decisão para obter um modelo capaz de detectar os clientes insatisfeitos com uma sensibilidade de cerca de 75%, mantendo a especificidade no mesmo nível.

Esse modelo simples, sendo facilmente interpretável, poderia ser usado sem dificuldade para apoiar as decisões de que ações tomar para manter a fidelidade dos clientes.
