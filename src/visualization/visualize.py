import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import networkx as nx
from tabulate import tabulate

class DataVisualization:
    @staticmethod
    def boxplot_plot(dataframe, column):
        """
        Plota um BoxPlot para a coluna especificada do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna: Nome da coluna para a qual deseja gerar o BoxPlot.
        """
        if (column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column].dtype)):
            plt.figure(figsize=(8, 6))
            plt.boxplot(dataframe[column], vert=False)
            plt.title(f'BoxPlot para a coluna "{column}"')
            plt.xlabel('Valores')
            plt.show()
        else:
            print(f'A coluna "{column}" não existe no DataFrame ou não é do tipo numérico.')
    
    @staticmethod
    def plotar_histogramas(dataframe, colunas_numericas):
        """
        Plota histogramas para as variáveis numéricas especificadas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param colunas_numericas: Lista de nomes das colunas numéricas para as quais deseja gerar os histogramas.
        """
        num_colunas = len(colunas_numericas)
        fig, axs = plt.subplots(nrows=num_colunas, ncols=1, figsize=(10, 5 * num_colunas))
        plt.subplots_adjust(hspace=0.5)  # Ajustar o espaço entre os subplots

        for i, coluna in enumerate(colunas_numericas):
            if coluna in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[coluna].dtype):
                axs[i].hist(dataframe[coluna], bins=30, color='skyblue', edgecolor='black')
                axs[i].set_title(f'Histograma para "{coluna}"')
                axs[i].set_xlabel('Valores')
                axs[i].set_ylabel('Frequência')

        plt.show()

    @staticmethod
    def plotar_dispersao(dataframe, coluna_x, coluna_y):
        """
        Plota um Gráfico de Dispersão para visualizar a relação entre duas variáveis.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna_x: Nome da coluna para o eixo x.
        :param coluna_y: Nome da coluna para o eixo y.
        """
        if coluna_x in dataframe.columns and coluna_y in dataframe.columns:
            plt.figure(figsize=(10, 6))

            if isinstance(dataframe[coluna_x].dtype, CategoricalDtype):
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe, hue=coluna_x)
            else:
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe)

            plt.title(f'Gráfico de Dispersão entre "{coluna_x}" e "{coluna_y}"')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.show()
        else:
            print(f'Uma ou ambas as colunas especificadas não existem no DataFrame.')

    @staticmethod
    def plotar_heatmap(dataframe):
        """
        Gera um Mapa de Calor (Heatmap) para visualizar a correlação entre as variáveis numéricas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        """
        # Seleciona apenas as variáveis numéricas
        variaveis_numericas = dataframe.select_dtypes(include='number')

        # Calcula a matriz de correlação
        matriz_correlacao = variaveis_numericas.corr()

        # Configuração do tamanho da figura
        plt.figure(figsize=(12, 8))

        # Cria o heatmap utilizando Seaborn
        sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f")

        # Adiciona título ao mapa de calor
        plt.title('Mapa de Calor - Correlação entre Variáveis Numéricas')

        # Exibe o mapa de calor
        plt.show()

    @staticmethod
    def plotar_grafico_linhas(dataframe, coluna_x, coluna_y):
        """
        Plota um gráfico de linhas para duas colunas específicas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna_x: Nome da coluna para o eixo x.
        :param coluna_y: Nome da coluna para o eixo y.
        """
        if coluna_x in dataframe.columns and coluna_y in dataframe.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(dataframe[coluna_x], dataframe[coluna_y], marker='o', linestyle='-')
            plt.title(f'Gráfico de Linhas para {coluna_y} em relação a {coluna_x}')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.grid(True)
            plt.show()
        else:
            print(f'Uma ou ambas as colunas ({coluna_x}, {coluna_y}) não existem no DataFrame.')
            
    @staticmethod
    def plotar_tabela_contingencia(dataframe, *colunas_categoricas):
        """
        Plota uma tabela de contingência para analisar a relação entre variáveis categóricas.

        :param dataframe: DataFrame a ser visualizado.
        :param colunas_categoricas: Nomes das colunas categóricas.
        """
        if len(colunas_categoricas) < 2:
            print("Forneça pelo menos duas colunas categóricas para criar uma tabela de contingência.")
            return

        tabela_contingencia = pd.crosstab(dataframe[colunas_categoricas[0]], dataframe[colunas_categoricas[1]])
        print("Tabela de Contingência:")
        print(tabela_contingencia)

        # Teste qui-quadrado
        chi2, p, _, _ = chi2_contingency(tabela_contingencia)
        print(f"\nResultado do Teste Qui-Quadrado:")
        print(f"Chi2: {chi2}")
        print(f"P-valor: {p}")

        # Plotar heatmap da tabela de contingência
        plt.figure(figsize=(10, 6))
        sns.heatmap(tabela_contingencia, annot=True, cmap='coolwarm', fmt="d")
        plt.title(f'Relação entre {", ".join(colunas_categoricas)}')
        plt.show()
        

    @staticmethod
    def plot_centralidade_histogram(centralidade_result, metric_name):
        """
        Plota um histograma para os resultados da centralidade.

        :param centralidade_result: Dicionário ou lista contendo os resultados da centralidade.
        :param metric_name: Nome da métrica de centralidade.
        """
        if isinstance(centralidade_result, dict):
            values = list(centralidade_result.values())
        elif isinstance(centralidade_result, list):
            values = centralidade_result
        else:
            raise ValueError("centralidade_result deve ser uma lista ou um dicionário.")

        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, edgecolor='black')
        plt.title(f'Histograma para {metric_name}')
        plt.xlabel(f'Valores de {metric_name}')
        plt.ylabel('Frequência')
        plt.show()

    @staticmethod
    def plot_graph(graph, pos=None):
        """
        Plota o grafo.

        :param graph: Grafo NetworkX.
        :param pos: Posições dos nós no gráfico (opcional).
        """
        plt.figure(figsize=(10, 8))
        if pos is None:
            pos = nx.spring_layout(graph)  # Você pode escolher um layout diferente se preferir.
        nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue')
        plt.title('Visualização do Grafo')
        plt.show()

    @staticmethod
    def plot_scaled_centralidade_histogram(centralidade_result, metric_name):
        """
        Plota um histograma para os resultados da centralidade, escalando os valores entre 0 e 1.

        :param centralidade_result: Dicionário contendo os resultados da centralidade.
        :param metric_name: Nome da métrica de centralidade.
        """
        values = list(centralidade_result.values())
        values_scaled = MinMaxScaler().fit_transform(np.array(values).reshape(-1, 1))
        plt.figure(figsize=(8, 6))
        plt.hist(values_scaled, bins=20, edgecolor='black')
        plt.title(f'Histograma Normalizado para {metric_name}')
        plt.xlabel(f'Valores Normalizados de {metric_name}')
        plt.ylabel('Frequência')
        plt.show()
        

    @staticmethod
    def plot_shortest_path(shortest_path_result, metric_name):
        """
        Plota um gráfico para o resultado do shortest path.

        :param shortest_path_result: Lista ou dicionário contendo o caminho mais curto.
        :param metric_name: Nome da métrica (por exemplo, "Shortest Path").
        """
        plt.figure(figsize=(8, 6))
        
        if isinstance(shortest_path_result, list):
            plt.plot(shortest_path_result, marker='o')
        elif isinstance(shortest_path_result, dict):
            nodes = list(shortest_path_result.keys())
            values = list(shortest_path_result.values())
            plt.bar(nodes, values)
        
        plt.title(f'Gráfico para {metric_name}')
        plt.xlabel('Índice do Nó')
        plt.ylabel(f'Valor de {metric_name}')
        plt.show()

        
    @staticmethod
    def plot_closeness_centrality(closeness_centrality_result, metric_name):
        """
        Plota um gráfico para a Closeness Centrality.

        :param closeness_centrality_result: Dicionário contendo os resultados da Closeness Centrality.
        :param metric_name: Nome da métrica (por exemplo, "Closeness Centrality").
        """
        nodes = list(closeness_centrality_result.keys())
        values = list(closeness_centrality_result.values())
        
        plt.figure(figsize=(10, 8))
        plt.bar(nodes, values)
        plt.title(f'Barra para {metric_name}')
        plt.xlabel('Índice do Nó')
        plt.ylabel(f'Valor de {metric_name}')
        plt.show()

    @staticmethod
    def plot_centralidade_efficiency(centralidade_result, metric_name):
        """
        Plota um gráfico para a eficiência.

        :param centralidade_result: Valor único representando a eficiência.
        :param metric_name: Nome da métrica (por exemplo, "Efficiency").
        """
        plt.figure(figsize=(6, 6))
        plt.bar([metric_name], [centralidade_result], color='skyblue')
        plt.title(f'Barra para {metric_name}')
        plt.xlabel('Métrica')
        plt.ylabel(f'Valor de {metric_name}')
        plt.show()
    
    @staticmethod    
    def plot_graph_erdos_small(graph, title):
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph, seed=42)  # Usando spring layout para posicionar os nós
        labels = nx.get_edge_attributes(graph, 'relationship')

        # Desenha nós
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue', alpha=0.8)

        # Desenha arestas
        nx.draw_networkx_edges(graph, pos, edge_color='gray')

        # Adiciona rótulos aos nós
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

        # Adiciona rótulos às arestas
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red')

        plt.title(title)
        plt.axis('off')  # Desativa os eixos
        plt.show()
        
    @staticmethod
    def efficiency_curve_erdos_small(graph):
        densities = []
        efficiencies = []

        # Varia a densidade de 0.1 a 1.0
        for density in range(1, 11):
            density /= 10.0
            densities.append(density)

            # Cria um grafo com a densidade específica
            g = nx.fast_gnp_random_graph(len(graph.nodes), density)
            efficiency = nx.global_efficiency(g)
            efficiencies.append(efficiency)

        return densities, efficiencies

    @staticmethod
    def plot_efficiency_curve_erdos_small(densities, efficiencies):
        plt.figure(figsize=(10, 6))
        plt.plot(densities, efficiencies, marker='o', linestyle='-', color='b')
        plt.title('Curva de Eficiência em Função da Densidade de Arestas')
        plt.xlabel('Densidade de Arestas')
        plt.ylabel('Eficiência Global')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_clustering_coefficient_erdos_small(graph, title):
        degrees = dict(graph.degree())
        clustering_coefficients = nx.clustering(graph)

        # Filtra nós isolados, pois o coeficiente de agrupamento não é definido para eles
        nodes = [node for node, degree in degrees.items() if degree > 0]
        clustering_values = [clustering_coefficients[node] for node in nodes]

        # Calcula o grau médio em cada ponto para um melhor entendimento
        average_degrees = {}
        for degree in set(degrees.values()):
            nodes_with_degree = [node for node, node_degree in degrees.items() if node_degree == degree]
            avg_clustering = sum(clustering_coefficients[node] for node in nodes_with_degree) / len(nodes_with_degree)
            average_degrees[degree] = avg_clustering

        plt.figure(figsize=(10, 6))
        plt.scatter(nodes, clustering_values, c='b', marker='o', label='Coeficiente de Agrupamento')
        plt.scatter(list(average_degrees.keys()), list(average_degrees.values()), c='r', marker='s', label='Grau Médio')

        plt.title(title)
        plt.xlabel('Grau do Nó')
        plt.ylabel('Coeficiente de Agrupamento')
        plt.legend()
        plt.show()
        

    def plot_er_graph(self, n_vals, p_vals):
        """
        Gera e exibe gráficos Erdos-Renyi para diferentes valores de N e P.

        Parameters:
        - n_vals: Tupla de valores de N.
        - p_vals: Lista de tuplas de valores de P.
        """
        sns.reset_defaults()
        sns.set_theme(rc={'figure.dpi': 72, 'savefig.dpi': 300,
                          'figure.autolayout': True})
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, ax = plt.subplots(len(n_vals), len(p_vals[0]), figsize=(15, 10))

        for ni, n in enumerate(n_vals):
            for pi, p in enumerate(p_vals[ni]):
                g = nx.erdos_renyi_graph(n=n, p=p)
                nx.draw_kamada_kawai(g, node_size=7, ax=ax[ni][pi])
                ax[ni][pi].set_title(f"N={n}, p={p:.2f}")

        fig.savefig('../reports/figures/er_graph.png')
        plt.show()