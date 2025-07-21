# Implementação de técnicas de transformada de Laplace inversa em relaxometria por Ressonância Magnética Nuclear (RMN) no domínio do tempo

![Python](https://img.shields.io/badge/python-3.10-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/status-completo-green)


Este projeto foi desenvolvido como parte de uma iniciação científica durante minha graduação em Licenciatura em Física na Universidade Federal de São Carlos, campus Araras. O objetivo era simular o decaimento multi-exponencial de um pulso de RMN, adicionar ruído branco ao sinal e, por fim, aplicar a transformada de Laplace inversa para obter a distribuição de tempos de relaxamento.

## Objetivos

## Fundamentos teóricos

## Tecnologias e bibliotecas utilizadas

- Python 2.x
- Jupyter Notebook
- Numpy
- Matplotlib
- Biblioteca personalizada 'Laplin', de autoria do meu orientador, professor João Teles de Carvalho Neto
- Dados experimentais provenientes de ensaios de laboratório

## Funcionalidades

- Geração de sinais simulados de decaimentos em RMN
- Adição de ruído branco com amplitude controlada
- Aplicação da transformada de Laplace inversa para extração de distribuições de tempo
- Visualizações gráficas dos sinais simulados e da distribuição resultante

## Estrutura

- 'notebooks/': contém os notebooks Jupyter com os experimentos e simulações
- 'data/': arquivos CSV com dados reais e simulados
- 'laplin/': implementação da biblioteca laplin (com permissão do autor para disponibilização)
- 'reports/': relatório final de pesquisa
- 'requirements.txt': dependências do projeto

## Como executar

1. Clone este repositório
2. (Opcional) Crie um ambiente virtual
3. Instale as dependências:
   '''bash
   pip install -r requirements.txt'''
4. Abra o Jupyter Notebook e execute os arquivos da pasta '/notebooks'

## Licença

Este projeto está licenciado sob a licença ...

## Autor

João J Ambrozetto
Projeto realizado como parte do programa de iniciação Científica .... UFSCar 2018-2019
