import pandas as pd

class AHP:
    """
    Classe para implementar o método AHP.

    Args:
        dataframe: Dataframe com as informações das bolsas de estudo.
        critérios: Lista com os critérios a serem avaliados.

    Methods:
        definir_hierarquia(): Define a hierarquia de critérios.
        definir_relações_de_importância(): Define as relações de importância entre os critérios.
        calcular_pesos_dos_critérios(): Calcula os pesos dos critérios.
        comparar_bolsas_de_estudo(): Compara duas bolsas de estudo com base nos pesos dos critérios.
    """

    def __init__(self, dataframe, criterios):
        self.dataframe = dataframe
        self.criterios = criterios
        self.pesos_dos_criterios = {}

    def definir_hierarquia(self):
        """
        Define a hierarquia de critérios.

        Returns:
            Hierarquia de critérios.
        """

        hierarquia = {}
        for criterio in self.criterios:
            hierarquia[criterio] = {}

        return hierarquia

    def definir_relacoes_de_importancia(self, hierarquia):
        """
        Define as relações de importância entre os critérios.

        Args:
            hierarquia: Hierarquia de critérios.

        Returns:
            Hierarquia de critérios com as relações de importância definidas.
        """

        for criterio1 in hierarquia:
            for criterio2 in hierarquia:
                if criterio1 != criterio2:
                    importancia = input(f"Qual a importância de {criterio1} em relação a {criterio2}?")
                    hierarquia[criterio1][criterio2] = importancia

        return hierarquia
    
    def calcular_pesos_dos_criterios(self, hierarquia):
        """
        Calcula os pesos dos critérios.

        Args:
            hierarquia: Hierarquia de critérios com as relações de importância definidas.

        Returns:
            Pesos dos critérios.
        """

        pesos = {}
        for criterio in hierarquia:
            pesos[criterio] = 0

        for criterio1 in hierarquia:
            for criterio2 in hierarquia[criterio1]:
                pesos[criterio1] += float(hierarquia[criterio1][criterio2]) / 2

        self.pesos_dos_criterios = pesos

    def comparar_bolsas_de_estudo(self, bolsa1, bolsa2):
        """
        Compara duas bolsas de estudo com base nos pesos dos critérios.

        Args:
            bolsa1: Bolsa de estudo 1.
            bolsa2: Bolsa de estudo 2.

        Returns:
            Índice de concordância entre as duas bolsas de estudo.
        """
        indice_de_concordancia = 0
        for criterio in self.criterios:
            if criterio not in bolsa1.index:
                raise KeyError(f"Coluna '{criterio}' não encontrada.")
            try:
                bolsa1_preferencia = float(bolsa1[criterio])
                bolsa2_preferencia = float(bolsa2[criterio])
            except ValueError:
                # Se não for possível converter para float, atribui um valor que não influenciará no cálculo
                bolsa1_preferencia = 0
                bolsa2_preferencia = 0

            indice_de_concordancia += self.pesos_dos_criterios[criterio] * (
                bolsa1_preferencia - bolsa2_preferencia
            )
        return indice_de_concordancia / (sum(self.pesos_dos_criterios))


