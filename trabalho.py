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
from skfuzzy import control as ctrl
import skfuzzy.control.visualization as viewer


# função de apoio para complemento fuzzy
def comp1(self,v):
    result = np.array()
    for i in v:
        result.append(1.-i)
    return result

# ANTECEDENTES

faixa_atraso = np.arange(1,50,0.1)
faixa_capacidade = np.arange(1,1000,1)
faixa_ocupacao = np.arange(0,100,0.1)
faixa_descarte = np.arange(0,100,0.1)

# funções de ajuste das entradas
atraso_baixo = fuzz.trimf(faixa_atraso, [1.,1.,5.])
atraso_medio = fuzz.trapmf(faixa_atraso, [2.,5.,20.,30.])
atraso_alto = fuzz.trapmf(faixa_atraso, [20.,30.,50.,50.])

capacidade_baixa = fuzz.trimf(faixa_capacidade, [1,1,50])
capacidade_media = fuzz.trapmf(faixa_capacidade, [10,20,200,300])
capacidade_alta = fuzz.trapmf(faixa_capacidade, [200,300,1000,1000])

ocupacao_baixa = fuzz.trimf(faixa_ocupacao, [0.,0.,30.])
ocupacao_media = fuzz.trapmf(faixa_ocupacao, [20.,30.,70.,80.])
ocupacao_alta = fuzz.trimf(faixa_ocupacao, [70.,100.,100.])

descarte_baixo = fuzz.trimf(faixa_descarte, [0.,0.,2.])
descarte_medio = fuzz.trapmf(faixa_descarte, [1.,2.,10.,11.])
descarte_alto = fuzz.trapmf(faixa_descarte, [5.,11.,100.,100.])

# Plota funções de pertinência dos antecedentes
fig_scale_x = 2.0
fig_scale_y = 1.5
fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
row = 2
col = 2
ax = plt.subplot(row, col, 1, ymargin=0.05, xmargin=0.05, xlabel='Atraso[ms]')
plt.title("Atraso associado ao enlace")
plt.ylabel('Pertinência')
plt.plot(faixa_atraso, atraso_baixo, label="Baixo")
plt.plot(faixa_atraso, atraso_medio, label="Médio")
plt.plot(faixa_atraso, atraso_alto, label="Alto")
plt.legend(loc="upper left")
plt.subplot(row, col, 2, ymargin=0.1, xmargin=0.1)
plt.title("Capacidade do enlace")
plt.xlabel("Mbps")
plt.ylabel('Pertinência')
plt.plot(faixa_capacidade, capacidade_baixa, label="Baixa")
plt.plot(faixa_capacidade, capacidade_media, label="Média")
plt.plot(faixa_capacidade, capacidade_alta, label="Alta")
plt.legend(loc="upper left")
plt.subplot(row, col, 3, ymargin=0.1, xmargin=0.1)
plt.title("Ocupação associada ao enlace")
plt.xlabel("Percentual")
plt.ylabel('Pertinência')
plt.plot(faixa_ocupacao, ocupacao_baixa, label="Baixa")
plt.plot(faixa_ocupacao, ocupacao_media, label="Média")
plt.plot(faixa_ocupacao, ocupacao_alta, label="Alta")
plt.legend(loc="upper left")
plt.subplot(row, col, 4, ymargin=0.1, xmargin=0.1)
plt.title("Descarte de pacotes associado ao enlace")
plt.xlabel("Percentual")
plt.ylabel('Pertinência')
plt.plot(faixa_descarte, descarte_baixo, label="Baixo")
plt.plot(faixa_descarte, descarte_medio, label="Médio")
plt.plot(faixa_descarte, descarte_alto, label="Alto")
plt.legend(loc="upper left")
#plt.show(block=True)

# entradas para Sugeno
while True:
    in_atraso = float(input("Informe o atraso do enlace, entre 1.0 e 50.0 (ms):"))
    if ((in_atraso<1.0) or (in_atraso>50.0)):
        print("O valor informado deve estar entre 1.0 e 50.0")
        continue
    break

while True:
    in_capacidade = int(input("Informe a capacidade do enlace, entre 1 e 1000 (Mbps):"))
    if ((in_capacidade<0.1) or (in_capacidade>1000)):
        print("O valor informado deve estar entre 1 e 1000")
        continue
    break

while True:
    in_ocupacao = float(input("Informe a ocupação do enlace (%), de 0.0 a 100:"))
    if ((in_ocupacao<0.0) or (in_ocupacao>100.0)):
        print("O valor informado deve estar entre 0.0 e 100.0")
        continue
    break

while True:
    in_descarte = float(input("Informe a taxa de descarte de pacotes (%), de 0.0 a 100:"))
    if ((in_descarte<0.0) or (in_descarte>100.0)):
        print("O valor informado deve estar entre 0.0 e 100.0")
        continue
    break

# fuzzyficação das entradas nítidas
atraso_nota_baixa = fuzz.interp_membership(faixa_atraso,atraso_baixo,in_atraso)
atraso_nota_media = fuzz.interp_membership(faixa_atraso,atraso_medio,in_atraso)
atraso_nota_alta = fuzz.interp_membership(faixa_atraso,atraso_alto,in_atraso)

capacidade_nota_baixa = fuzz.interp_membership(faixa_capacidade, capacidade_baixa, in_capacidade)
capacidade_nota_media = fuzz.interp_membership(faixa_capacidade, capacidade_media, in_capacidade)
capacidade_nota_alta = fuzz.interp_membership(faixa_capacidade, capacidade_alta,  in_capacidade)

ocupacao_nota_baixa = fuzz.interp_membership(faixa_ocupacao,ocupacao_baixa,in_ocupacao)
ocupacao_nota_media = fuzz.interp_membership(faixa_ocupacao,ocupacao_media,in_ocupacao)
ocupacao_nota_alta = fuzz.interp_membership(faixa_ocupacao,ocupacao_alta,in_ocupacao)

descarte_nota_baixa = fuzz.interp_membership(faixa_descarte,descarte_baixo,in_descarte)
descarte_nota_media = fuzz.interp_membership(faixa_descarte,descarte_medio,in_descarte)
descarte_nota_alta = fuzz.interp_membership(faixa_descarte,descarte_alto,in_descarte)

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




# topologia de avaliação


# para permitir a visualização dos gráficos
plt.show(block=True)