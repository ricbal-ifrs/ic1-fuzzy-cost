#!/usr/bin/python
#coding: utf-8

#################################################################
# Controlador fuzzy para determinar custos de enlaces no problema
# de determinar o melhor caminho em uma rede
# Referente a trabalho final do bloco 01 da disciplina IC1
# Semestre: abr/2025
# Autor: Ricardo Balbinot
# Versões:
# 0.1 - abr/2025 - Ricardo Balbinot
#  - Proposta do controlador
#
# TODO:
# - Realizar testes do controlador com medições reais da rede
#   simulada no Mininet e avaliar impacto das decisões tomadas
#################################################################

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import networkx as nx
import copy
import random
import scipy.stats as stats
import time
import skfuzzy.control.visualization as viewer



# função de apoio para complemento fuzzy
def comp1(self,v):
    result = np.array()
    for i in v:
        result.append(1.-i)
    return result

class fuzzyDelayCost:
    def __init__(self):
        """
        Inicializa a controladora fuzzy para determinar peso de enlaces
        favorecendo serviços sensíveis ao atraso
        """
        # ANTECEDENTES
        self.faixa_atraso = np.arange(1,50,0.1)
        self.faixa_capacidade = np.arange(1,1000,1)
        self.faixa_ocupacao = np.arange(0,100,0.1)
        self.faixa_descarte = np.arange(0,100,0.1)
        # funções de pertinência das entradas
        self.atraso_baixo = fuzz.trimf(self.faixa_atraso, [1.,1.,5.])
        self.atraso_medio = fuzz.trapmf(self.faixa_atraso, [2.,5.,20.,30.])
        self.atraso_alto = fuzz.trapmf(self.faixa_atraso, [20.,30.,50.,50.])
        self.capacidade_baixa = fuzz.trimf(self.faixa_capacidade, [1,1,50])
        self.capacidade_media = fuzz.trapmf(self.faixa_capacidade, [10,20,200,300])
        self.capacidade_alta = fuzz.trapmf(self.faixa_capacidade, [200,300,1000,1000])
        self.ocupacao_baixa = fuzz.trimf(self.faixa_ocupacao, [0.,0.,30.])
        self.ocupacao_media = fuzz.trapmf(self.faixa_ocupacao, [20.,30.,70.,80.])
        self.ocupacao_alta = fuzz.trimf(self.faixa_ocupacao, [70.,100.,100.])
        self.descarte_baixo = fuzz.trimf(self.faixa_descarte, [0.,0.,2.])
        self.descarte_medio = fuzz.trapmf(self.faixa_descarte, [1.,2.,10.,11.])
        self.descarte_alto = fuzz.trapmf(self.faixa_descarte, [5.,11.,100.,100.])

    def plotarPertinencias(self):
        """
        Elabora gráfico das funções de pertinência das entradas

        Retorna: 
          matplotlib.figure-- figura com gráficos solicitados
        """
        # Plota funções de pertinência dos antecedentes
        fig_scale_x = 2.0
        fig_scale_y = 1.5
        fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
        row = 2
        col = 2
        ax = plt.subplot(row, col, 1, ymargin=0.05, xmargin=0.05, xlabel='Atraso[ms]')
        plt.title("Atraso associado ao enlace")
        plt.ylabel('Pertinência')
        plt.plot(self.faixa_atraso, self.atraso_baixo, label="Baixo")
        plt.plot(self.faixa_atraso, self.atraso_medio, label="Médio")
        plt.plot(self.faixa_atraso, self.atraso_alto, label="Alto")
        plt.legend(loc="upper left")
        plt.subplot(row, col, 2, ymargin=0.1, xmargin=0.1)
        plt.title("Capacidade do enlace")
        plt.xlabel("Mbps")
        plt.ylabel('Pertinência')
        plt.plot(self.faixa_capacidade, self.capacidade_baixa, label="Baixa")
        plt.plot(self.faixa_capacidade, self.capacidade_media, label="Média")
        plt.plot(self.faixa_capacidade, self.capacidade_alta, label="Alta")
        plt.legend(loc="upper left")
        plt.subplot(row, col, 3, ymargin=0.1, xmargin=0.1)
        plt.title("Ocupação associada ao enlace")
        plt.xlabel("Percentual")
        plt.ylabel('Pertinência')
        plt.plot(self.faixa_ocupacao, self.ocupacao_baixa, label="Baixa")
        plt.plot(self.faixa_ocupacao, self.ocupacao_media, label="Média")
        plt.plot(self.faixa_ocupacao, self.ocupacao_alta, label="Alta")
        plt.legend(loc="upper left")
        plt.subplot(row, col, 4, ymargin=0.1, xmargin=0.1)
        plt.title("Descarte de pacotes associado ao enlace")
        plt.xlabel("Percentual")
        plt.ylabel('Pertinência')
        plt.plot(self.faixa_descarte, self.descarte_baixo, label="Baixo")
        plt.plot(self.faixa_descarte, self.descarte_medio, label="Médio")
        plt.plot(self.faixa_descarte, self.descarte_alto, label="Alto")
        plt.legend(loc="upper left")
        return fig

    def calculaCusto(self,atraso,capacidade,ocupacao,descarte):
        """
        Determina o custo em razão das entradas nítidas e avaliação
        do controlador fuzzy

        Parâmetros:
            atraso (float): atraso em milisegundos, limitado na faixa 1 a 50ms
            capacidade (int): capacidade, em Mpbs, limitada na faixa de 1 a 1000Mbps
            ocupacao (float): percentual de ocupação do enlace, de 0 a 100%
            descarte (float): percentual de descarte de pacotes observado no enlace, de 0 a 100%

        Retorna:
            float: valor calculado do custo
        """
        # fuzzyficação das entradas nítidas
        atraso_nota_baixa = fuzz.interp_membership(self.faixa_atraso, self.atraso_baixo, atraso)
        atraso_nota_media = fuzz.interp_membership(self.faixa_atraso, self.atraso_medio, atraso)
        atraso_nota_alta = fuzz.interp_membership(self.faixa_atraso, self.atraso_alto, atraso)

        capacidade_nota_baixa = fuzz.interp_membership(self.faixa_capacidade, self.capacidade_baixa, capacidade)
        capacidade_nota_media = fuzz.interp_membership(self.faixa_capacidade, self.capacidade_media, capacidade)
        capacidade_nota_alta = fuzz.interp_membership(self.faixa_capacidade, self.capacidade_alta,  capacidade)

        ocupacao_nota_baixa = fuzz.interp_membership(self.faixa_ocupacao, self.ocupacao_baixa, ocupacao)
        ocupacao_nota_media = fuzz.interp_membership(self.faixa_ocupacao, self.ocupacao_media, ocupacao)
        ocupacao_nota_alta = fuzz.interp_membership(self.faixa_ocupacao, self.ocupacao_alta, ocupacao)

        descarte_nota_baixa = fuzz.interp_membership(self.faixa_descarte, self.descarte_baixo, descarte)
        descarte_nota_media = fuzz.interp_membership(self.faixa_descarte, self.descarte_medio, descarte)
        descarte_nota_alta = fuzz.interp_membership(self.faixa_descarte, self.descarte_alto, descarte)

        # Relata as avaliações de pertinência
        print("Atraso - baixo: %e" % atraso_nota_baixa)
        print("Atraso - médio: %e" % atraso_nota_media)
        print("Atraso - alto: %e" % atraso_nota_alta)
        print("Capacidade - baixa: %e" % capacidade_nota_baixa)
        print("Capacidade - média: %e" % capacidade_nota_media)
        print("Capacidade - alta: %e" % capacidade_nota_alta)
        print("Ocupação - baixa: %e" % ocupacao_nota_baixa)
        print("Ocupação - média: %e" % ocupacao_nota_media)
        print("Ocupação - alta: %e" % ocupacao_nota_alta)
        print("Descarte - baixo: %e" % descarte_nota_baixa)
        print("Descarte - médio: %e" % descarte_nota_media)
        print("Descarte - alto: %e" % descarte_nota_alta)
        # regras (Sugeno)
        # Custo priorizando atraso baixo
        # AND = fmin
        # OR = fmax
        # ATRASO = baixo
        # SE atraso = baixo E ocupacao = (baixa ou media) E descarte=baixo ENTAO CustoDelay = f1
        w1A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w1B = np.fmin(w1A,descarte_nota_baixa)
        w1 = np.fmin(atraso_nota_baixa,w1B)
        f1 =  9*(0.7*(1-atraso_nota_baixa)+0.3*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media)+(1-descarte_nota_baixa))/3)+1
        print("Regra 1: %e - Consequência estimada: %f" % (w1,f1))
        # SE atraso=baixo E capacidade = (média ou alta) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f2
        w2A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w2B = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w2C = np.fmin(w2A,w2B)
        w2 = np.fmin(atraso_nota_baixa,w2C)
        f2 =  10*(0.5*(1-atraso_nota_baixa)+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+7*ocupacao_nota_alta+
            (1-descarte_nota_baixa))/10)+10
        print("Regra 2: %e - Consequência estimada: %f" % (w2,f2))
        # SE atraso=baixo E capacidade = (baixa) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f3
        w3A = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w3B = np.fmin(capacidade_nota_baixa,w2A)
        w3 = np.fmin(atraso_nota_baixa,w3B)
        f3 =  10*(0.5*(1-atraso_nota_baixa)+0.5*(4*capacidade_nota_baixa+5*ocupacao_nota_alta+
            (1-descarte_nota_baixa)/10))+20
        print("Regra 3: %e - Consequência estimada: %f" % (w3,f3))
        # SE atraso = baixo E capacidade = (any) E ocupacao = (baixa ou media) E descarte=médio ENTAO CustoDelay = f4
        w4A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w4B = np.fmin(w4A,descarte_nota_media)
        w4 = np.fmin(atraso_nota_baixa,w4B)
        f4 = 20*(0.5*(1-atraso_nota_baixa)+0.5*(2*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media))+8*descarte_nota_media)/10)+10
        print("Regra 4: %e - Consequência estimada: %f" % (w4,f4))
        # SE atraso=baixo E capacidade = (média ou alta) E ocupacao = (alta) E descarte = médio ENTAO CustoDelay = f5
        w5A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w5B = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w5C = np.fmin(w5A,w5B)
        w5 = np.fmin(atraso_nota_baixa,w5C)
        f5 =  20*(0.5*(1-atraso_nota_baixa)+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+4*ocupacao_nota_alta+
            4*descarte_nota_baixa)/10)+20
        print("Regra 5: %e - Consequência estimada: %f" % (w5,f5))
        # SE atraso=baixo E capacidade = (baixa) E ocupacao = (alta) E descarte = medio ENTAO CustoDelay = f6
        w6A = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w6B = np.fmin(capacidade_nota_baixa,w6A)
        w6 = np.fmin(atraso_nota_baixa,w6B)
        f6 =  20*(0.5*(1-atraso_nota_baixa)+0.5*(2*capacidade_nota_baixa+4*ocupacao_nota_alta+
            4*descarte_nota_media)/10)+30
        print("Regra 6: %e - Consequência estimada: %f" % (w6,f6))
        # SE atraso = baixo E capacidade = (any) E ocupacao = (baixa,media ou alta) E descarte=alto ENTAO CustoDelay = f7
        w7A = np.fmax(ocupacao_nota_baixa,np.fmax(ocupacao_nota_media,ocupacao_nota_alta))
        w7B = np.fmin(w7A,descarte_nota_alta)
        w7 = np.fmin(atraso_nota_baixa,w7B)
        f7 =  30*(0.4*(1-atraso_nota_baixa)+0.6*(4*((1-ocupacao_nota_baixa)+
            ocupacao_nota_media+ocupacao_nota_alta)+6*descarte_nota_alta)/10)+60
        print("Regra 7: %e - Consequência estimada: %f" % (w7,f7))
        # ATRASO = médio
        # SE atraso = médio E ocupacao = (baixa ou media) E descarte=baixo ENTAO CustoDelay = f8
        w8A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w8B = np.fmin(w8A,descarte_nota_baixa)
        w8 = np.fmin(atraso_nota_media,w8B)
        f8 =  10*(0.7*atraso_nota_media+0.3*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media)+(1-descarte_nota_baixa))/3)+10
        print("Regra 8: %e - Consequência estimada: %f" % (w8,f8))
        # SE atraso=médio E capacidade = (média ou alta) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f9
        w9A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w9B = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w9C = np.fmin(w9A,w9B)
        w9 = np.fmin(atraso_nota_media,w9C)
        f9 =  10*(0.5*atraso_nota_media+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+7*ocupacao_nota_alta+
            (1-descarte_nota_baixa))/10)+20
        print("Regra 9: %e - Consequência estimada: %f" % (w9,f9))
        # SE atraso=médio E capacidade = (baixa) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f10
        w10A = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w10B = np.fmin(capacidade_nota_baixa,w10A)
        w10 = np.fmin(atraso_nota_media,w10B)
        f10 =  10*(0.5*atraso_nota_media+0.5*(4*capacidade_nota_baixa+5*ocupacao_nota_alta+
            (1-descarte_nota_baixa)/10))+30
        print("Regra 10: %e - Consequência estimada: %f" % (w10,f10))
        # SE atraso = médio E capacidade = (any) E ocupacao = (baixa ou media) E descarte=médio ENTAO CustoDelay = f11
        w11A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w11B = np.fmin(w11A,descarte_nota_media)
        w11 = np.fmin(atraso_nota_media,w11B)
        f11 = 20*(0.5*atraso_nota_media+0.5*(2*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media))+8*descarte_nota_media)/10)+20
        print("Regra 11: %e - Consequência estimada: %f" % (w11,f11))
        # SE atraso=médio E capacidade = (média ou alta) E ocupacao = (alta) E descarte = médio ENTAO CustoDelay = f12
        w12A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w12B = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w12C = np.fmin(w12A,w12B)
        w12 = np.fmin(atraso_nota_media,w12C)
        f12 =  20*(0.5*atraso_nota_media+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+4*ocupacao_nota_alta+
            4*descarte_nota_baixa)/10)+30
        print("Regra 12: %e - Consequência estimada: %f" % (w12,f12))
        # SE atraso=médio E capacidade = (baixa) E ocupacao = (alta) E descarte = medio ENTAO CustoDelay = f13
        w13A = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w13B = np.fmin(capacidade_nota_baixa,w13A)
        w13 = np.fmin(atraso_nota_media,w13B)
        f13 =  20*(0.5*atraso_nota_media+0.5*(2*capacidade_nota_baixa+4*ocupacao_nota_alta+
            4*descarte_nota_media)/10)+40
        print("Regra 13: %e - Consequência estimada: %f" % (w13,f13))
        # SE atraso = médio E capacidade = (any) E ocupacao = (baixa,media ou alta) E descarte=alto ENTAO CustoDelay = f14
        w14A = np.fmax(ocupacao_nota_baixa,np.fmax(ocupacao_nota_media,ocupacao_nota_alta))
        w14B = np.fmin(w14A,descarte_nota_alta)
        w14 = np.fmin(atraso_nota_media,w14B)
        f14 =  30*(0.4*atraso_nota_media+0.6*(4*((1-ocupacao_nota_baixa)+
            ocupacao_nota_media+ocupacao_nota_alta)+6*descarte_nota_alta)/10)+70
        print("Regra 14: %e - Consequência estimada: %f" % (w14,f14))
        # ATRASO = alto
        # SE atraso = alto EENTAO CustoDelay = f15
        w15 = atraso_nota_alta
        f15 =  20*(0.6*atraso_nota_alta+0.4*(2*capacidade_nota_baixa+4*((1-ocupacao_nota_baixa)+ocupacao_nota_media+ocupacao_nota_alta)+
                                    4*((1-descarte_nota_baixa)+descarte_nota_media+descarte_nota_alta)))/10+80
        print("Regra 15: %e - Consequência estimada: %f" % (w15,f15))
        custo_delay = (w1*f1+w2*f2+w3*f3+w4*f4+w5*f5+w6*f6+w7*f7+w8*f8+w9*f9+w10*f10+w11*f11+w12*f12+w13*f13+
               w14*f14+w15*f15)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15)
        print("Custo final: %f" % custo_delay)
        return custo_delay

# altera o peso dos enlaces da topologia indicada, de acordo com a categoria de serviço e 
# o controlador fuzzy

def calcula_pesos(topo,service,fuzzy):
    """
    Determina os pesos dos enlaces na topologia, de acordo com as regras do controlador
    fuzzy e a categoria de serviço indicada

    Parâmetros:
        topo (nx.Graph): grafo representando a topologia sob teste
        service (int): categoria de serviço desejada
    Retorna:
        nx.Graph: nova topologia, com pesos definidos para uso
    """
    nova = copy.deepcopy(topo)
    for link in nova.edges:
        atraso = nova[link[0]][link[1]]["latency"]
        capacidade = nova[link[0]][link[1]]['capacity']
        ocupacao = nova[link[0]][link[1]]['occupation']
        descarte = nova[link[0]][link[1]]['drop']
        custo = fuzzy.calculaCusto(atraso,capacidade,ocupacao,descarte)
        nova[link[0]][link[1]]['weight'] = custo
        print('Enlace: ',str(link))
        print('\tCusto: %f'%custo)
    return nova


# aplica djikstra para encontrar as melhores árvores entre todos os pares (origem,destino) possíveis

def calcula_caminhos(topo):
    """
    Calcula os caminhos de acordo com o custo definido pelo controlador fuzzy
    para os enlaces.
    
    TODO: nessa versão para simplificar a visualização, consideramos a rede simétrica
    ou seja, um melhor caminho (src,dst) é o mesmo melhor caminho (dst,src)
    Necessário rever isso em casos assimétricos.
    """
    caminhos = []
    analisados = []
    # determina todos os caminhos (origem,destino) via Djikstra
    for src in topo.nodes:
        # src tem que ser conectável
        if topo.degree(src)==0:
            continue
        # src agora é o nó de origem
        for dst in topo.nodes:
            # destino tem que ser conectável
            if (topo.degree(dst)==0):
                continue
            if (src==dst):
                continue
            pathA = (src,dst)
            pathB = (dst,src)
            if (pathA in analisados):
                continue
            if (pathB in analisados):
                continue
            analisados.append(pathA)
            path = nx.dijkstra_path(topo,src,dst)
            caminhos.append(path)
    return caminhos

def drawTopo(topo):
    pos = nx.spring_layout(topo,seed=101291211)
    fig, ax = plt.subplots()
    nx.draw_networkx_edges(
        topo,
        pos = pos,
        ax = ax,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(topo,'weight')
    for key in labels:
        formatado = "{:.2f}".format(labels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        topo,
        edge_labels = labels,
        pos = pos,
        ax = ax
    )
    nx.draw_networkx_nodes(
        topo,
        pos = pos,
        ax = ax,
        node_size=1500
    )
    labels = nx.get_node_attributes(topo,'name')
    nx.draw_networkx_labels(
        topo,
        labels = labels,
        pos = pos,
        ax = ax
    )
    return fig,ax


def drawPath(topo,caminho):
    local = copy.deepcopy(topo)
    # verifica se os nós da árvore estão no caminho 
    # e remove todos que não fazem parte
    for nodo in list(local.nodes):
        remover = True
        for passo in caminho:
            if (nodo==passo):
                remover = False
        if remover:
            local.remove_node(nodo)
    pos = nx.spring_layout(topo,seed=101291211)
    fig, ax = plt.subplots()
    nx.draw_networkx_edges(
        local,
        pos = pos,
        ax = ax,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(local,'weight')
    for key in labels:
        formatado = "{:.2f}".format(labels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        local,
        edge_labels = labels,
        pos = pos,
        ax = ax
    )
    nx.draw_networkx_nodes(
        local,
        pos = pos,
        ax = ax,
        node_size=1500
    )
    labels = nx.get_node_attributes(local,'name')
    nx.draw_networkx_labels(
        local,
        labels = labels,
        pos = pos,
        ax = ax
    )
    return fig,ax

def calcula_metrica(topo,caminhos):
    # quero percorrer cada caminho e totalizar, por caminho escolhido
    # os dados de atraso, descarte, ocupacao e capacidade
    atrasos = []
    maximo_atrasos = []
    minimo_atrasos = []
    descartes = []
    ocupacoes = []
    capacidades = []
    for path in caminhos:
        atraso = 0
        max_atraso = 0
        min_atraso = 100
        descarte = 0
        ocupacao = 0
        capacidade = 0
        src = None
        dst = None
        for passo in path:
            # constrói adjascencia
            if src is None:
                src = passo
                continue
            else:
                if dst is None:
                    dst = passo
                else: 
                    src = dst
                    dst = passo
            x = topo[src][dst]['latency']
            atraso += x
            if (x>max_atraso):
                max_atraso=x
            if (x<min_atraso):
                min_atraso=x
            x = topo[src][dst]['drop']
            if x>descarte:
                descarte=x
            x = topo[src][dst]['occupation']
            if x>ocupacao:
                ocupacao=x
            x = topo[src][dst]['capacity']
            if x>capacidade:
                capacidade=x
        atrasos.append(atraso)
        maximo_atrasos.append(max_atraso)
        minimo_atrasos.append(min_atraso)
        descartes.append(descarte)
        ocupacoes.append(ocupacao)
        capacidades.append(capacidade)
    return atrasos, maximo_atrasos, minimo_atrasos, descartes, ocupacoes, capacidades

class topoA:
    def __init__(self):
        """
        Cria o modelo de topologia A para avaliação.
        Tipo: Hub and spoke (HS)
        """
        self.topo = nx.Graph(nome='Topologia A')
        # adiciona os nós e enlaces entre todos os nós
        for i in range(1,11):
            peso = 1
            uso = 1
            nome = 'SW'+str(i)
            self.topo.add_node(i,weight=peso,use=uso,name=nome)
        # agora adiciona os enlaces de forma a criar a topologia
        peso = 1.0
        atraso = 1.0
        ocupacao = 1.0
        capacidade = 100
        descarte = 0.0
        self.topo.add_edge(1,7,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,2,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,3,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,5,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,7,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,5,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(6,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
    
    def randomizeStats(self,seed,max_delay,max_occup,max_drop):
        """
        Gera status aleatórios dos enlaces a partir da semente fornecida.
        Garante o mesmo comportamento dada a mesma semente.

        Args:
            seed (int): Semente do gerador
        """
        random.seed(seed)
        # percorre todos os nós e gera os caminhos
        for link in self.topo.edges:
            src = link[0]
            dst = link[1]
            atraso = random.uniform(1,max_delay)
            ocupacao = random.uniform(0,max_occup)
            descarte = random.uniform(0,max_drop)
            self.topo[src][dst]['latency'] = atraso
            self.topo[src][dst]['occupation'] = ocupacao
            self.topo[src][dst]['drop'] = descarte
            #print(self.topo[src][dst])


    def describeTopo(self):
        """
        Retorna os parâmetros de característica da topologia
        """
        N = self.topo.number_of_nodes()
        M = self.topo.number_of_edges()
        sumD = 0
        max_degree = 0
        for node in self.topo.nodes:
            grau = self.topo.degree(node)
            sumD += grau
            if (grau>max_degree):
                max_degree=grau
        mean_degree = sumD/N
        sumR = 0
        for node in self.topo.nodes:
            if self.topo.degree(node)>mean_degree:
                sumR+=1
        R = sumR/N
        G = max_degree
        GR = G/N
        return N,M,mean_degree,G,R,GR
    
    def __str__(self):
        N,M,mean_degree,G,R,GR = self.describeTopo()
        desc = 'TopoA -- N: {:d}, M: {:d}, g: {:.2f}, G: {:.2f}, R: {:.2f}, Gr: {:.2f}'
        desc = desc.format(N,M,mean_degree,G,R,GR)
        return desc

def plotaBarras(dados1,dados2,dados3,dados4,titulo,ylabel):
    d1 = np.array(dados1)
    media1 = np.mean(d1)
    std1 = np.std(d1)
    maximo1 = np.max(d1)
    minimo1 = np.min(d1)
    d2 = np.array(dados2)
    media2 = np.mean(d2)
    std2 = np.std(d2)
    maximo2 = np.max(d2)
    minimo2 = np.min(d2)
    d3 = np.array(dados3)
    media3 = np.mean(d3)
    std3 = np.std(d3)
    maximo3 = np.max(d3)
    minimo3 = np.min(d3)
    d4 = np.array(dados4)
    media4 = np.mean(d4)
    std4 = np.std(d4)
    maximo4 = np.max(d4)
    minimo4 = np.min(d4)

    media = [media1, media2, media3, media4]
    std = [std1, std2, std3, std4]
    maximo = [maximo1, maximo2, maximo3, maximo4]
    minimo = [minimo1, minimo2, minimo3, minimo4]

    categorias = ['Média', 'Desvio Padrão', 'Máximo', 'Mínimo']
    grupos = ['Djikstra puro', 'Djikstra por atraso', 'Djikstra ponderado', 'Fuzzy']

    # Organizando os dados por categoria
    dados = [media, std, maximo, minimo]

    # Configuração do gráfico
    x = np.arange(len(categorias))  # posições das categorias
    largura = 0.2  # largura das barras

    # Criando as barras
    fig, ax = plt.subplots()
    ax.bar(x - largura, [d[0] for d in dados], width=largura, label=grupos[0])
    ax.bar(x, [d[1] for d in dados], width=largura, label=grupos[1])
    ax.bar(x + largura, [d[2] for d in dados], width=largura, label=grupos[2])
    ax.bar(x + 2*largura, [d[3] for d in dados], width=largura, label=grupos[3])

    # Rótulos e título
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.legend()

    plt.tight_layout()

# # agora adiciona os enlaces entre todos os nós
# for nA in topo1.nodes:
#     for nB in topo1.nodes:
#         if (nA!=nB):
#             if ((not topo1.has_edge(nA,nB))or(not topo1.has_edge(nB,nA))):
#                 peso = 1
#                 atraso = 1.0
#                 ocupacao = 1.0
#                 capacidade = 100
#                 descarte = 0.1
#                 topo1.add_edge(nA,nB,weight=peso,latency=atraso,occupation=ocupacao,drop=descarte)


delay = fuzzyDelayCost()
topo1 = topoA()
print(topo1)
topo1.randomizeStats(14532,7,95,1.5)
drawTopo(topo1.topo)

# agora tem que comparar os métodos
# djikstra simples, por distância
caminhos1 = calcula_caminhos(topo1.topo)
atrasos1, maximo_atrasos1, minimo_atrasos1, descartes1, ocupacoes1, capacidades1= calcula_metrica(topo1.topo,caminhos1)

# djikstra com custo por atraso puro
ctopo = copy.deepcopy(topo1.topo)
for link in ctopo.edges:
    src=link[0]
    dst=link[1]
    ctopo[src][dst]['weight'] = ctopo[src][dst]['latency']
caminhos2 = calcula_caminhos(ctopo)
atrasos2, maximo_atrasos2, minimo_atrasos2, descartes2, ocupacoes2, capacidades2= calcula_metrica(ctopo,caminhos2)

# djikstra com custo formulado
ctopo = copy.deepcopy(topo1.topo)
for link in ctopo.edges:
    src=link[0]
    dst=link[1]
    ctopo[src][dst]['weight'] = 0.4*ctopo[src][dst]['latency']+0.3*ctopo[src][dst]['occupation']+0.3*ctopo[src][dst]['drop']
caminhos3 = calcula_caminhos(ctopo)
atrasos3, maximo_atrasos3, minimo_atrasos3, descartes3, ocupacoes3, capacidades3= calcula_metrica(ctopo,caminhos3)

# método proposto
novatopo = calcula_pesos(topo1.topo,1,delay)
drawTopo(novatopo)
caminhos4 = calcula_caminhos(novatopo)
atrasos4, maximo_atrasos4, minimo_atrasos4, descartes4, ocupacoes4, capacidades4= calcula_metrica(novatopo,caminhos4)

# Atrasos
plotaBarras(atrasos1,atrasos2,atrasos3,atrasos4,'Atrasos acumulados','milisegundos')
plotaBarras(maximo_atrasos1,maximo_atrasos2,maximo_atrasos3,maximo_atrasos4,'Atrasos máximos','milisegundos')
plotaBarras(minimo_atrasos1,minimo_atrasos2,minimo_atrasos3,minimo_atrasos4,'Atrasos mínimos','milisegundos')
plotaBarras(ocupacoes1,ocupacoes2,ocupacoes3,ocupacoes4,'Ocupações','%')
plotaBarras(descartes1,descartes2,descartes3,descartes4,'Descartes','%')

print('ATRASOS CUMULATIVOS')
print("DJIKSTRA PURO")
print(stats.describe(np.array(atrasos1)))
print("DJIKSTRA POR ATRASO")
print(stats.describe(np.array(atrasos2)))
print("DJIKSTRA PONDERADO")
print(stats.describe(np.array(atrasos3)))
print("DJIKSTRA FUZZY")
print(stats.describe(np.array(atrasos4)))

print('MÁXIMOS ATRASOS')
print("DJIKSTRA PURO")
print(stats.describe(np.array(maximo_atrasos1)))
print("DJIKSTRA POR ATRASO")
print(stats.describe(np.array(maximo_atrasos2)))
print("DJIKSTRA PONDERADO")
print(stats.describe(np.array(maximo_atrasos3)))
print("DJIKSTRA FUZZY")
print(stats.describe(np.array(maximo_atrasos4)))

print('MÍNIMOS ATRASOS')
print("DJIKSTRA PURO")
print(stats.describe(np.array(minimo_atrasos1)))
print("DJIKSTRA POR ATRASO")
print(stats.describe(np.array(minimo_atrasos2)))
print("DJIKSTRA PONDERADO")
print(stats.describe(np.array(minimo_atrasos3)))
print("DJIKSTRA FUZZY")
print(stats.describe(np.array(minimo_atrasos4)))

print('OCUPAÇÕES')
print("DJIKSTRA PURO")
print(stats.describe(np.array(ocupacoes1)))
print("DJIKSTRA POR ATRASO")
print(stats.describe(np.array(ocupacoes2)))
print("DJIKSTRA PONDERADO")
print(stats.describe(np.array(ocupacoes3)))
print("DJIKSTRA FUZZY")
print(stats.describe(np.array(ocupacoes4)))

print('DESCARTES')
print("DJIKSTRA PURO")
print(stats.describe(np.array(descartes1)))
print("DJIKSTRA POR ATRASO")
print(stats.describe(np.array(descartes2)))
print("DJIKSTRA PONDERADO")
print(stats.describe(np.array(descartes3)))
print("DJIKSTRA FUZZY")
print(stats.describe(np.array(descartes4)))

#for path in caminhos:
     #drawPath(novatopo,path)



# agora aplica o controlador para determinar as melhores árvores por atraso entre os pares (origem,destino)

# para permitir a visualização dos gráficos
plt.show(block=True)


# entradas para Sugeno
# while True:
#     in_atraso = float(input("Informe o atraso do enlace, entre 1.0 e 50.0 (ms):"))
#     if ((in_atraso<1.0) or (in_atraso>50.0)):
#         print("O valor informado deve estar entre 1.0 e 50.0")
#         continue
#     break

# while True:
#     in_capacidade = int(input("Informe a capacidade do enlace, entre 1 e 1000 (Mbps):"))
#     if ((in_capacidade<0.1) or (in_capacidade>1000)):
#         print("O valor informado deve estar entre 1 e 1000")
#         continue
#     break

# while True:
#     in_ocupacao = float(input("Informe a ocupação do enlace (%), de 0.0 a 100:"))
#     if ((in_ocupacao<0.0) or (in_ocupacao>100.0)):
#         print("O valor informado deve estar entre 0.0 e 100.0")
#         continue
#     break

# while True:
#     in_descarte = float(input("Informe a taxa de descarte de pacotes (%), de 0.0 a 100:"))
#     if ((in_descarte<0.0) or (in_descarte>100.0)):
#         print("O valor informado deve estar entre 0.0 e 100.0")
#         continue
#     break