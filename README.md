# Implementação de técnicas de transformada de Laplace inversa em relaxometria por Ressonância Magnética Nuclear (RMN) no domínio do tempo

![Python 2.7](https://img.shields.io/badge/python-2.7-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/status-completo-green)


Este projeto foi desenvolvido como parte de uma iniciação científica durante minha graduação em Licenciatura em Física na Universidade Federal de São Carlos, campus Araras. O objetivo era simular o decaimento multi-exponencial de um pulso de RMN, adicionar ruído branco ao sinal e, por fim, aplicar a transformada de Laplace inversa para obter a distribuição de tempos de relaxamento.

## Objetivos

Estudar, implementar e validar métodos computacionais de Transformada de Laplace Inversa (TLI) para a obtenção de distribuições de tempos de relaxação em Ressonância Magnética Nuclear no Domínio do Tempo (RMN-DT).

Como objetivos específicos, destacamos:

- Aprendizagem dos conceitos básicos de RMN e, principalmente, de RMN-DT.
- Domínio básico do uso de um espectrômetro de RMN-DT de baixo campo, além da capacidade de ajustar parâmetros relevantes para as sequências de pulso mais importantes para os estudos de relaxação como, por exemplo, CPMG.
- Estudo das propriedades básicas da Transformada de Laplace Inversa e estudo dos principais métodos computacionais para a implementação da TLI.
- Implementação experimental de um ou mais métodos de TLI objetivando a obtenção das curas de distribuição dos tempos de relaxação para amostras modelo.

## Fundamentos teóricos

## Tecnologias e bibliotecas utilizadas

- Python 2.7
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

## Autores

João J Ambrozetto
Projeto realizado como parte do programa de iniciação Científica .... UFSCar 2018-2019

João Teles de Carvalho Neto
Professor do Departamento de Ciências da Natureza e Matemática DCNME
Ufscar Araras
