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
import matplotlib.gridspec as gridspec
import networkx as nx
import copy
import random
import scipy.stats as stats
import logging 
import skfuzzy.control.visualization as viewer

#################################################################
# Arquivos de log
#################################################################
# para console e para arquivo....
formato = logging.Formatter('%(name)s: %(asctime)s - [%(levelname)s] -- %(message)s',
                datefmt='%d/%m/%Y-%H:%M:%S')
ch1 = logging.StreamHandler()
ch1.setLevel(logging.INFO)
ch1.setFormatter(formato)
ch2 = logging.FileHandler('saidas.log',mode='w')
ch2.setLevel(logging.INFO)
ch2.setFormatter(formato)
# cria o logger 
logfuzzy = logging.getLogger('[FUZZY]')
logfuzzy.addHandler(ch1)
logfuzzy.addHandler(ch2)
logtestes = logging.getLogger('[TESTES]')
logtestes.addHandler(ch1)
logtestes.addHandler(ch2)

#################################################################
# Controlador fuzzy para determinação de custo de enlace
#################################################################
class fuzzyDelayCost:
    def __init__(self):
        """
        Inicializa a controladora fuzzy para determinar peso de enlaces
        favorecendo serviços sensíveis ao atraso
        """
        logfuzzy.info('Iniciando controlador fuzzy')
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
        logfuzzy.debug("Atraso - baixo: %e" % atraso_nota_baixa)
        logfuzzy.debug("Atraso - médio: %e" % atraso_nota_media)
        logfuzzy.debug("Atraso - alto: %e" % atraso_nota_alta)
        logfuzzy.debug("Capacidade - baixa: %e" % capacidade_nota_baixa)
        logfuzzy.debug("Capacidade - média: %e" % capacidade_nota_media)
        logfuzzy.debug("Capacidade - alta: %e" % capacidade_nota_alta)
        logfuzzy.debug("Ocupação - baixa: %e" % ocupacao_nota_baixa)
        logfuzzy.debug("Ocupação - média: %e" % ocupacao_nota_media)
        logfuzzy.debug("Ocupação - alta: %e" % ocupacao_nota_alta)
        logfuzzy.debug("Descarte - baixo: %e" % descarte_nota_baixa)
        logfuzzy.debug("Descarte - médio: %e" % descarte_nota_media)
        logfuzzy.debug("Descarte - alto: %e" % descarte_nota_alta)
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
        logfuzzy.debug("Regra 1: %e - Consequência estimada: %f" % (w1,f1))
        # SE atraso=baixo E capacidade = (média ou alta) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f2
        w2A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w2B = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w2C = np.fmin(w2A,w2B)
        w2 = np.fmin(atraso_nota_baixa,w2C)
        f2 =  10*(0.5*(1-atraso_nota_baixa)+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+7*ocupacao_nota_alta+
            (1-descarte_nota_baixa))/10)+10
        logfuzzy.debug("Regra 2: %e - Consequência estimada: %f" % (w2,f2))
        # SE atraso=baixo E capacidade = (baixa) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f3
        w3A = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w3B = np.fmin(capacidade_nota_baixa,w2A)
        w3 = np.fmin(atraso_nota_baixa,w3B)
        f3 =  10*(0.5*(1-atraso_nota_baixa)+0.5*(4*capacidade_nota_baixa+5*ocupacao_nota_alta+
            (1-descarte_nota_baixa)/10))+20
        logfuzzy.debug("Regra 3: %e - Consequência estimada: %f" % (w3,f3))
        # SE atraso = baixo E capacidade = (any) E ocupacao = (baixa ou media) E descarte=médio ENTAO CustoDelay = f4
        w4A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w4B = np.fmin(w4A,descarte_nota_media)
        w4 = np.fmin(atraso_nota_baixa,w4B)
        f4 = 20*(0.5*(1-atraso_nota_baixa)+0.5*(2*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media))+8*descarte_nota_media)/10)+10
        logfuzzy.debug("Regra 4: %e - Consequência estimada: %f" % (w4,f4))
        # SE atraso=baixo E capacidade = (média ou alta) E ocupacao = (alta) E descarte = médio ENTAO CustoDelay = f5
        w5A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w5B = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w5C = np.fmin(w5A,w5B)
        w5 = np.fmin(atraso_nota_baixa,w5C)
        f5 =  20*(0.5*(1-atraso_nota_baixa)+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+4*ocupacao_nota_alta+
            4*descarte_nota_baixa)/10)+20
        logfuzzy.debug("Regra 5: %e - Consequência estimada: %f" % (w5,f5))
        # SE atraso=baixo E capacidade = (baixa) E ocupacao = (alta) E descarte = medio ENTAO CustoDelay = f6
        w6A = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w6B = np.fmin(capacidade_nota_baixa,w6A)
        w6 = np.fmin(atraso_nota_baixa,w6B)
        f6 =  20*(0.5*(1-atraso_nota_baixa)+0.5*(2*capacidade_nota_baixa+4*ocupacao_nota_alta+
            4*descarte_nota_media)/10)+30
        logfuzzy.debug("Regra 6: %e - Consequência estimada: %f" % (w6,f6))
        # SE atraso = baixo E capacidade = (any) E ocupacao = (baixa,media ou alta) E descarte=alto ENTAO CustoDelay = f7
        w7A = np.fmax(ocupacao_nota_baixa,np.fmax(ocupacao_nota_media,ocupacao_nota_alta))
        w7B = np.fmin(w7A,descarte_nota_alta)
        w7 = np.fmin(atraso_nota_baixa,w7B)
        f7 =  30*(0.4*(1-atraso_nota_baixa)+0.6*(4*((1-ocupacao_nota_baixa)+
            ocupacao_nota_media+ocupacao_nota_alta)+6*descarte_nota_alta)/10)+60
        logfuzzy.debug("Regra 7: %e - Consequência estimada: %f" % (w7,f7))
        # ATRASO = médio
        # SE atraso = médio E ocupacao = (baixa ou media) E descarte=baixo ENTAO CustoDelay = f8
        w8A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w8B = np.fmin(w8A,descarte_nota_baixa)
        w8 = np.fmin(atraso_nota_media,w8B)
        f8 =  10*(0.7*atraso_nota_media+0.3*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media)+(1-descarte_nota_baixa))/3)+10
        logfuzzy.debug("Regra 8: %e - Consequência estimada: %f" % (w8,f8))
        # SE atraso=médio E capacidade = (média ou alta) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f9
        w9A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w9B = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w9C = np.fmin(w9A,w9B)
        w9 = np.fmin(atraso_nota_media,w9C)
        f9 =  10*(0.5*atraso_nota_media+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+7*ocupacao_nota_alta+
            (1-descarte_nota_baixa))/10)+20
        logfuzzy.debug("Regra 9: %e - Consequência estimada: %f" % (w9,f9))
        # SE atraso=médio E capacidade = (baixa) E ocupacao = (alta) E descarte = baixo ENTAO CustoDelay = f10
        w10A = np.fmin(ocupacao_nota_alta,descarte_nota_baixa)
        w10B = np.fmin(capacidade_nota_baixa,w10A)
        w10 = np.fmin(atraso_nota_media,w10B)
        f10 =  10*(0.5*atraso_nota_media+0.5*(4*capacidade_nota_baixa+5*ocupacao_nota_alta+
            (1-descarte_nota_baixa)/10))+30
        logfuzzy.debug("Regra 10: %e - Consequência estimada: %f" % (w10,f10))
        # SE atraso = médio E capacidade = (any) E ocupacao = (baixa ou media) E descarte=médio ENTAO CustoDelay = f11
        w11A = np.fmax(ocupacao_nota_baixa,ocupacao_nota_media)
        w11B = np.fmin(w11A,descarte_nota_media)
        w11 = np.fmin(atraso_nota_media,w11B)
        f11 = 20*(0.5*atraso_nota_media+0.5*(2*((1-ocupacao_nota_baixa)+(1-ocupacao_nota_media))+8*descarte_nota_media)/10)+20
        logfuzzy.debug("Regra 11: %e - Consequência estimada: %f" % (w11,f11))
        # SE atraso=médio E capacidade = (média ou alta) E ocupacao = (alta) E descarte = médio ENTAO CustoDelay = f12
        w12A = np.fmax(capacidade_nota_media,capacidade_nota_alta)
        w12B = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w12C = np.fmin(w12A,w12B)
        w12 = np.fmin(atraso_nota_media,w12C)
        f12 =  20*(0.5*atraso_nota_media+0.5*((1-capacidade_nota_media)+(1-capacidade_nota_alta)+4*ocupacao_nota_alta+
            4*descarte_nota_baixa)/10)+30
        logfuzzy.debug("Regra 12: %e - Consequência estimada: %f" % (w12,f12))
        # SE atraso=médio E capacidade = (baixa) E ocupacao = (alta) E descarte = medio ENTAO CustoDelay = f13
        w13A = np.fmin(ocupacao_nota_alta,descarte_nota_media)
        w13B = np.fmin(capacidade_nota_baixa,w13A)
        w13 = np.fmin(atraso_nota_media,w13B)
        f13 =  20*(0.5*atraso_nota_media+0.5*(2*capacidade_nota_baixa+4*ocupacao_nota_alta+
            4*descarte_nota_media)/10)+40
        logfuzzy.debug("Regra 13: %e - Consequência estimada: %f" % (w13,f13))
        # SE atraso = médio E capacidade = (any) E ocupacao = (baixa,media ou alta) E descarte=alto ENTAO CustoDelay = f14
        w14A = np.fmax(ocupacao_nota_baixa,np.fmax(ocupacao_nota_media,ocupacao_nota_alta))
        w14B = np.fmin(w14A,descarte_nota_alta)
        w14 = np.fmin(atraso_nota_media,w14B)
        f14 =  30*(0.4*atraso_nota_media+0.6*(4*((1-ocupacao_nota_baixa)+
            ocupacao_nota_media+ocupacao_nota_alta)+6*descarte_nota_alta)/10)+70
        logfuzzy.debug("Regra 14: %e - Consequência estimada: %f" % (w14,f14))
        # ATRASO = alto
        # SE atraso = alto EENTAO CustoDelay = f15
        w15 = atraso_nota_alta
        f15 =  20*(0.6*atraso_nota_alta+0.4*(2*capacidade_nota_baixa+4*((1-ocupacao_nota_baixa)+ocupacao_nota_media+ocupacao_nota_alta)+
                                    4*((1-descarte_nota_baixa)+descarte_nota_media+descarte_nota_alta)))/10+80
        logfuzzy.debug("Regra 15: %e - Consequência estimada: %f" % (w15,f15))
        custo_delay = (w1*f1+w2*f2+w3*f3+w4*f4+w5*f5+w6*f6+w7*f7+w8*f8+w9*f9+w10*f10+w11*f11+w12*f12+w13*f13+
               w14*f14+w15*f15)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15)
        logfuzzy.debug("Custo final: %f" % custo_delay)
        return custo_delay


#################################################################
# Funções de apoio
#################################################################

# função de apoio para complemento fuzzy
def comp1(v):
    """
    Determina o complemento de um conjunto fuzzy

    Parâmetros:
        v: lista de valores

    Returns:
        lista: lista complementada
    """
    result = np.array()
    for i in v:
        result.append(1.-i)
    return result

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
        logfuzzy.info('Enlace: ',str(link))
        logfuzzy.info('\tCusto: %f'%custo)
    return nova

def calcula_caminhos(topo):
    """
    Calcula os caminhos de acordo com o custo definido pelo controlador fuzzy
    para os enlaces, usando o algoritmo de Djikstra, para todos os pares (origem,destino)
    
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
    """
    Gera uma representação gráfica da topologia.

    Parâmetros:
        topo (networkx.Graph): grafo da topologia

    Retorna:
        fig, ax: Referência para a figura e eixos da figura (matplotlib)
    """
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


def drawPath(topoA,caminhoA,topoB,caminhoB,topoC,caminhoC,topoD,caminhoD):
    """
    Gera uma representação gráfica dos caminhos associados a uma topologia

    Parâmetros:
        topo (networkx.Graph): grafo da topologia
        caminho (lista): identificadores dos nós pertecentes ao caminho
            entre par (origem,destino) na topologia indicada

    Retorna:
        fig, ax: Referência para a figura e eixos da figura (matplotlib)
    """
    local1 = copy.deepcopy(topoA)
    local2 = copy.deepcopy(topoB)
    local3 = copy.deepcopy(topoC)
    local4 = copy.deepcopy(topoD)
    #  apenas nodos do caminho
    local1 = nx.subgraph(local1,caminhoA)
    local2 = nx.subgraph(local2,caminhoB)
    local3 = nx.subgraph(local3,caminhoC)
    local4 = nx.subgraph(local4,caminhoD)
    first = caminhoA[0]
    last = caminhoA[-1]
    pos = nx.spring_layout(topoA,seed=101291211)
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('De: %d Para: %d' %(first,last))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Custo por número de hops\n'+str(caminhoA))
    # topologia 1
    nx.draw_networkx_edges(
        local1,
        pos = pos,
        ax = ax1,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(local1,'weight')
    llabels = nx.get_edge_attributes(local1,'latency')
    olabels = nx.get_edge_attributes(local1,'occupation')
    dlabels = nx.get_edge_attributes(local1,'drop')
    for key in labels:
        formatado = "(C:{:.2f})".format(labels[key])+ "\n(L:{:.2f})".format(llabels[key])+"-(O:{:.2f})".format(olabels[key])+"-(D:{:.2f})".format(dlabels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        local1,
        edge_labels = labels,
        pos = pos,
        ax = ax1
    )
    nx.draw_networkx_nodes(
        local1,
        pos = pos,
        ax = ax1,
        node_size=1500
    )
    labels = nx.get_node_attributes(local1,'name')
    nx.draw_networkx_labels(
        local1,
        labels = labels,
        pos = pos,
        ax = ax1
    )
    # topologia 2
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('Custo por latência\n'+str(caminhoB))
    nx.draw_networkx_edges(
        local2,
        pos = pos,
        ax = ax2,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(local2,'weight')
    llabels = nx.get_edge_attributes(local2,'latency')
    olabels = nx.get_edge_attributes(local2,'occupation')
    dlabels = nx.get_edge_attributes(local2,'drop')
    for key in labels:
        formatado = "(C:{:.2f})".format(labels[key])+ "\n(L:{:.2f})".format(llabels[key])+"-(O:{:.2f})".format(olabels[key])+"-(D:{:.2f})".format(dlabels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        local2,
        edge_labels = labels,
        pos = pos,
        ax = ax2
    )
    nx.draw_networkx_nodes(
        local2,
        pos = pos,
        ax = ax2,
        node_size=1500
    )
    labels = nx.get_node_attributes(local2,'name')
    nx.draw_networkx_labels(
        local2,
        labels = labels,
        pos = pos,
        ax = ax2
    )
    # topologia 3
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_title('Custo ponderado\n'+str(caminhoC))
    nx.draw_networkx_edges(
        local3,
        pos = pos,
        ax = ax3,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(local3,'weight')
    llabels = nx.get_edge_attributes(local3,'latency')
    olabels = nx.get_edge_attributes(local3,'occupation')
    dlabels = nx.get_edge_attributes(local3,'drop')
    for key in labels:
        formatado = "(C:{:.2f})".format(labels[key])+ "\n(L:{:.2f})".format(llabels[key])+"-(O:{:.2f})".format(olabels[key])+"-(D:{:.2f})".format(dlabels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        local3,
        edge_labels = labels,
        pos = pos,
        ax = ax3
    )
    nx.draw_networkx_nodes(
        local3,
        pos = pos,
        ax = ax3,
        node_size=1500
    )
    labels = nx.get_node_attributes(local3,'name')
    nx.draw_networkx_labels(
        local3,
        labels = labels,
        pos = pos,
        ax = ax3
    )
    # topologia 4
    ax4 = fig.add_subplot(gs[1,1])
    ax4.set_title('Custo fuzzy\n'+str(caminhoD))
    nx.draw_networkx_edges(
        local4,
        pos = pos,
        ax = ax4,
        arrows = True,
        arrowstyle='-',
        min_source_margin=5,
        min_target_margin=5,
    )
    labels = nx.get_edge_attributes(local4,'weight')
    llabels = nx.get_edge_attributes(local4,'latency')
    olabels = nx.get_edge_attributes(local4,'occupation')
    dlabels = nx.get_edge_attributes(local4,'drop')
    for key in labels:
        formatado = "(C:{:.2f})".format(labels[key])+ "\n(L:{:.2f})".format(llabels[key])+"-(O:{:.2f})".format(olabels[key])+"-(D:{:.2f})".format(dlabels[key])
        labels[key] = formatado
    nx.draw_networkx_edge_labels(
        local4,
        edge_labels = labels,
        pos = pos,
        ax = ax4
    )
    nx.draw_networkx_nodes(
        local4,
        pos = pos,
        ax = ax4,
        node_size=1500
    )
    labels = nx.get_node_attributes(local4,'name')
    nx.draw_networkx_labels(
        local4,
        labels = labels,
        pos = pos,
        ax = ax4
    )
    

def calcula_metrica(topo,caminhos):
    """
    Determina métricas de avaliação do comportamento dos caminhos na topologia
    indicada, em termos de atraso cumulativo do caminho, máximo atrasos percebidos,
    mínimos atrasos percebidos, taxa de descarte de pacotes, taxa de ocupação e 
    capacidade dos enlaces. 

    Parâmetros:
        topo (networkx.Graph): topologia da rede avaliada
        caminhos (lista de listas de nós): caminhos possíveis na topologia

    Retorna:
        atrasos, maximo_atrasos, minimo_atrasos, 
        descartes, ocupacoes, capacidades: listas com as métricas para cada caminho indicado
    """
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

def describeTopo(topo):
    """
    Retorna os parâmetros de característica da topologia

    Parâmetros:
        topo (networkx.Graph): grafo representativo da topologia

    Retorna:
        N,M,mean_degree,G,R,GR (float) - índices de avaliação da topologia
    """
    N = topo.number_of_nodes()
    M = topo.number_of_edges()
    sumD = 0
    max_degree = 0
    for node in topo.nodes:
        grau = topo.degree(node)
        sumD += grau
        if (grau>max_degree):
            max_degree=grau
    mean_degree = sumD/N
    sumR = 0
    for node in topo.nodes:
        if topo.degree(node)>mean_degree:
            sumR+=1
    R = sumR/N
    G = max_degree
    GR = G/N
    return N,M,mean_degree,G,R,GR

#################################################################
# Topologia de testes A - Hub and spoke
#################################################################

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

    def randomizeStats2(self,seed,max_delay,max_occup,max_drop):
        """
        Gera status aleatórios dos enlaces a partir da semente fornecida.
        Garante o mesmo comportamento dada a mesma semente.
        Neste segundo caso, garante que a ocupação tenha relação com um possível
        maior atraso no enlace. 

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

    def __str__(self):
        N,M,mean_degree,G,R,GR = describeTopo(self.topo)
        desc = 'TopoA -- N: {:d}, M: {:d}, g: {:.2f}, G: {:.2f}, R: {:.2f}, Gr: {:.2f}'
        desc = desc.format(N,M,mean_degree,G,R,GR)
        return desc

#################################################################
# Topologia de testes B - Star
#################################################################

class topoB:
    def __init__(self):
        """
        Cria o modelo de topologia A para avaliação.
        Tipo: Star
        """
        self.topo = nx.Graph(nome='Topologia B')
        # adiciona os nós e enlaces entre todos os nós
        for i in range(1,16):
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
        self.topo.add_edge(1,2,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,3,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,5,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,7,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,11,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,12,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,3,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,12,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,7,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,11,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(6,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(6,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(7,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(7,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(7,12,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(8,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(8,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(9,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(9,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(9,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(10,11,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(10,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(11,13,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(11,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(12,15,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(13,14,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
    
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

    def randomizeStats2(self,seed,max_delay,max_occup,max_drop):
        """
        Gera status aleatórios dos enlaces a partir da semente fornecida.
        Garante o mesmo comportamento dada a mesma semente.
        Neste segundo caso, garante que a ocupação tenha relação com um possível
        maior atraso no enlace. 

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

    def __str__(self):
        N,M,mean_degree,G,R,GR = describeTopo(self.topo)
        desc = 'TopoB -- N: {:d}, M: {:d}, g: {:.2f}, G: {:.2f}, R: {:.2f}, Gr: {:.2f}'
        desc = desc.format(N,M,mean_degree,G,R,GR)
        return desc

#################################################################
# Topologia de testes C - Ladder
#################################################################

class topoC:
    def __init__(self):
        """
        Cria o modelo de topologia A para avaliação.
        Tipo: Ladder
        """
        self.topo = nx.Graph(nome='Topologia C')
        # adiciona os nós e enlaces entre todos os nós
        for i in range(1,13):
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
        self.topo.add_edge(1,2,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,3,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(1,4,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,5,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,6,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(2,12,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(3,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,7,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(4,11,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(5,9,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(6,8,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(6,11,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(7,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(8,12,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
        self.topo.add_edge(9,10,weight=peso,latency=atraso,capacity=capacidade,occupation=ocupacao,drop=descarte)
    
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

    def randomizeStats2(self,seed,max_delay,max_occup,max_drop):
        """
        Gera status aleatórios dos enlaces a partir da semente fornecida.
        Garante o mesmo comportamento dada a mesma semente.
        Neste segundo caso, garante que a ocupação tenha relação com um possível
        maior atraso no enlace. 

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

    def __str__(self):
        N,M,mean_degree,G,R,GR = describeTopo(self.topo)
        desc = 'TopoC -- N: {:d}, M: {:d}, g: {:.2f}, G: {:.2f}, R: {:.2f}, Gr: {:.2f}'
        desc = desc.format(N,M,mean_degree,G,R,GR)
        return desc
    
#################################################################
# Funções auxiliares para testes e resultados
#################################################################

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

def runTeste(topo,id_topo,draw):
    """
    Roda um cenário de teste com a topologia indicada

    Args:
        topo (networkx.Graph): topologia sendo avaliada
        id_topo (std): string indentificando a topologia nos gráficos
        draw(bool): indica se os caminho do método proposto devem ser esboçados contra 
                    as demais técnicas
    """
    drawTopo(topo)
    # agora tem que comparar os métodos
    # djikstra simples, por distância
    caminhos1 = calcula_caminhos(topo)
    atrasos1, maximo_atrasos1, minimo_atrasos1, descartes1, ocupacoes1, capacidades1= calcula_metrica(topo,caminhos1)
    # djikstra com custo por atraso puro
    ctopo2 = copy.deepcopy(topo)
    for link in ctopo2.edges:
        src=link[0]
        dst=link[1]
        ctopo2[src][dst]['weight'] = ctopo2[src][dst]['latency']
    caminhos2 = calcula_caminhos(ctopo2)
    atrasos2, maximo_atrasos2, minimo_atrasos2, descartes2, ocupacoes2, capacidades2= calcula_metrica(ctopo2,caminhos2)
    # djikstra com custo formulado
    ctopo3 = copy.deepcopy(topo)
    for link in ctopo3.edges:
        src=link[0]
        dst=link[1]
        ctopo3[src][dst]['weight'] = 0.5*ctopo3[src][dst]['latency']+0.2*ctopo3[src][dst]['occupation']+0.3*ctopo3[src][dst]['drop']
    caminhos3 = calcula_caminhos(ctopo3)
    atrasos3, maximo_atrasos3, minimo_atrasos3, descartes3, ocupacoes3, capacidades3= calcula_metrica(ctopo3,caminhos3)
    # método proposto
    novatopo = calcula_pesos(topo,1,delay)
    drawTopo(novatopo)
    caminhos4 = calcula_caminhos(novatopo)
    atrasos4, maximo_atrasos4, minimo_atrasos4, descartes4, ocupacoes4, capacidades4= calcula_metrica(novatopo,caminhos4)
    # Atrasos
    desc = 'Atrasos acumulados ('+id_topo+')'
    plotaBarras(atrasos1,atrasos2,atrasos3,atrasos4,desc,'milisegundos')
    desc = 'Atrasos máximos ('+id_topo+')'
    plotaBarras(maximo_atrasos1,maximo_atrasos2,maximo_atrasos3,maximo_atrasos4,desc,'milisegundos')
    desc = 'Atrasos mínimos ('+id_topo+')'
    plotaBarras(minimo_atrasos1,minimo_atrasos2,minimo_atrasos3,minimo_atrasos4,desc,'milisegundos')
    desc = 'Ocupações ('+id_topo+')'
    plotaBarras(ocupacoes1,ocupacoes2,ocupacoes3,ocupacoes4,desc,'%')
    desc = 'Descartes ('+id_topo+')'
    plotaBarras(descartes1,descartes2,descartes3,descartes4,desc,'%')
    logtestes.info("*********************************************************")
    logtestes.info("TOPOLOGIA A")
    logtestes.info('ATRASOS CUMULATIVOS')
    logtestes.info("DJIKSTRA PURO")
    logtestes.info(stats.describe(np.array(atrasos1)))
    logtestes.info("DJIKSTRA POR ATRASO")
    logtestes.info(stats.describe(np.array(atrasos2)))
    logtestes.info("DJIKSTRA PONDERADO")
    logtestes.info(stats.describe(np.array(atrasos3)))
    logtestes.info("DJIKSTRA FUZZY")
    logtestes.info(stats.describe(np.array(atrasos4)))
    logtestes.info("************")
    logtestes.info('MÁXIMOS ATRASOS')
    logtestes.info("DJIKSTRA PURO")
    logtestes.info(stats.describe(np.array(maximo_atrasos1)))
    logtestes.info("DJIKSTRA POR ATRASO")
    logtestes.info(stats.describe(np.array(maximo_atrasos2)))
    logtestes.info("DJIKSTRA PONDERADO")
    logtestes.info(stats.describe(np.array(maximo_atrasos3)))
    logtestes.info("DJIKSTRA FUZZY")
    logtestes.info(stats.describe(np.array(maximo_atrasos4)))
    logtestes.info("************")
    logtestes.info('MÍNIMOS ATRASOS')
    logtestes.info("DJIKSTRA PURO")
    logtestes.info(stats.describe(np.array(minimo_atrasos1)))
    logtestes.info("DJIKSTRA POR ATRASO")
    logtestes.info(stats.describe(np.array(minimo_atrasos2)))
    logtestes.info("DJIKSTRA PONDERADO")
    logtestes.info(stats.describe(np.array(minimo_atrasos3)))
    logtestes.info("DJIKSTRA FUZZY")
    logtestes.info(stats.describe(np.array(minimo_atrasos4)))
    logtestes.info("************")
    logtestes.info('OCUPAÇÕES')
    logtestes.info("DJIKSTRA PURO")
    logtestes.info(stats.describe(np.array(ocupacoes1)))
    logtestes.info("DJIKSTRA POR ATRASO")
    logtestes.info(stats.describe(np.array(ocupacoes2)))
    logtestes.info("DJIKSTRA PONDERADO")
    logtestes.info(stats.describe(np.array(ocupacoes3)))
    logtestes.info("DJIKSTRA FUZZY")
    logtestes.info(stats.describe(np.array(ocupacoes4)))
    logtestes.info("************")
    logtestes.info('DESCARTES')
    logtestes.info("DJIKSTRA PURO")
    logtestes.info(stats.describe(np.array(descartes1)))
    logtestes.info("DJIKSTRA POR ATRASO")
    logtestes.info(stats.describe(np.array(descartes2)))
    logtestes.info("DJIKSTRA PONDERADO")
    logtestes.info(stats.describe(np.array(descartes3)))
    logtestes.info("DJIKSTRA FUZZY")
    logtestes.info(stats.describe(np.array(descartes4)))
    logtestes.info("*********************************************************")
    # desenha todos os caminhos alternativos encontrados
    if draw:
        for path1,path2,path3,path4 in zip(caminhos1,caminhos2,caminhos3,caminhos4):
            if (len(path1)==2 and len(path2)==2) and (len(path3)==2) and (len(path4)==2):
                # não desenha casos de caminhos diretos
                continue
            drawPath(topo,path1,ctopo2,path2,ctopo3,path3,novatopo,path4)

#################################################################
# Testes
#################################################################

delay = fuzzyDelayCost()

# topologia de testes A (Hub & Spoke)
topo1 = topoA()
print(topo1)
topo1.randomizeStats(14532,7,95,1.5)
runTeste(topo1.topo,'TopoA',False)

topo2 = topoB()
print(topo2)
topo2.randomizeStats(14532,7,95,1.5)
runTeste(topo2.topo,'TopoB',True)

topo3 = topoC()
print(topo3)
topo3.randomizeStats(14532,7,95,1.5)
runTeste(topo3.topo,'TopoC',False)

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