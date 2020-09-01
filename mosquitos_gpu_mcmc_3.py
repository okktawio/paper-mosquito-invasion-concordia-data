# coding: utf-8
import pymc
from numpy import *
import os, sys
from gpu_mosquitos_gl import gpu_base
import pylab


if 'CPU' in sys.argv: dispositivo = 'cpu'
else: dispositivo = 'gpu'

gpmosquitos = gpu_base(dispositivo)
like_mosquitos = gpmosquitos.like
mascaras = load('mascaras_x.npy')

def like_params(params):
    x0, y0, t0, radio, oo, pe, Kb, r, v, Db, per, mxr = params
    if Db < 0: Db = radio
    if mxr < 0:
        per = 0.0
        mxr = 1.0
    return like_mosquitos(D = radio ** 2 * pi,
                          r = r,
                          oo = oo,
                          pe = pe,
                          x0 = x0,
                          y0 = y0,
                          dinvasion = t0,
                          Kbaldio = Kb,
                          Dbaldio = Db ** 2 * pi,
                          per = per,
                          mxr = mxr,
                          v = v)


def guardalog(like, params, paramsn, acepta, deltas, c, b, a, nproceso = ''):
    log = open("mosquitos_gpu_mcmc_%s.log"%nproceso, 'a')
    log.write("%10.4f "%like)
    for i in params:
        log.write("%10.6f "%i)
    for i in paramsn:
        log.write("%10.6f "%i)
    for i in acepta:
        log.write("%10.6f "%i)
    for i in deltas:
        log.write("%10.6f "%i)
    log.write("%10.6f "%c)
    log.write("%10.6f "%b)
    log.write("%10.6f"%a)
    log.write("\n")
    log.close()
            
def gradiente_a(priors, nproceso, verbose = True):
    mosquitos = load("datos_mosquitos_x.npy")
    superior = array([495,  332,  500,   200.0,   100,  10.0,    1,     2.0,      2.0,      200,      -1.0,   -2])
    inferior = array([  0,    0,    0,     0.0,  -100,   0.0,    0,     0.0,     -2.0,        0,      -1.0,   -2])
    sd      = array([130.1995456351703,  148.2349794203014,  4.278484442888318,  1.6883612822486544,    0.009736898053259016,   0.00020962938446376196,    0.006072481368054586,      25.27215671649804,     8.895958354005033e-06,      1.0,      1.0,    8.304882410756425e-06])
    sd *= 4
    acepta = zeros_like(sd) + 0.25
    params = zeros_like(sd)
    paramsn = zeros_like(sd)
    paramsm = zeros_like(sd)
    try:
        tablas = loadtxt("mosquitos_gpu_mcmc_%s.log"%nproceso[0])
        todo = tablas[where(tablas[:,0] == tablas[:,0].max()),:][0][0]
        #print todo
        todo = todo[1:]
        #params  += todo[:12]
        params  += clip(todo[:12], inferior, superior)
        #params[9] = params[3]
        #print todo.shape, params.shape
        del(tablas)
        try:
            todo = loadtxt("mosquitos_gpu_mcmc_%s.log"%nproceso[1])[-1][1:]
            paramsn += clip(todo[12:24], inferior, superior)
            paramsm += clip(todo[:12], inferior, superior)
            sd[:]      = todo[24:36]
            #sd[:4] /= 100.0
            acepta[:]  = todo[36:48]
        except IOError: pass
        del(todo)
    except IOError:
        print("levantando de las priors")
        x0 = priors["x0"][0][0]
        y0 = priors["y0"][0][0]
        t0 = priors["t0"][0][0]
        radio = priors["radio"][0][0]
        r = priors["r"][0][0]
        oo = priors["oo"][0][0]
        pe = priors["pe"][0][0]
        Kb = priors["Kb"][0][0]
        Db = priors["Db"][0][0]
        per = 0.0
        mxr = 1.0
        v = priors["v"][0][0]
        params   = array([x0,     y0,   t0, radio,  oo,   pe,  Kb,  r,  v,  Db, per, mxr])
        
        params = clip(params, inferior, superior)
        paramsn = zeros_like(params) + params
        paramsm = zeros_like(params) + params


    if nproceso[0] != nproceso[1]:
        print('nprocesos distintos', nproceso)
        #params[0] += 44
        #params[1] += 55
        #params[7] = params[8]
        #params[8] = params[11]
        paramsm[:] = params[:]
        #sd[:2] = 1.

    set_printoptions(precision = 8, suppress = True)

    like0 = like_params(paramsm)
    if int(paramsm[0]) == 0: like0 -= 4
    if int(paramsm[1]) == 0: like0 -= 4
    j = -2
    for iteraciones in range(500):
        swt = iteraciones%50
        if swt:
            j += 1
            paramsn[:] = paramsm
            #if j >= 9: j = 11
            #if j == 7: j = 8
            paramsn[j] = params[j]

            try:
                while inferior[j] == superior[j]:
                    paramsn[j] = inferior[j]
                    params[j] = inferior[j]
                    j += 1
            except IndexError:
                j = -1
                continue
            while paramsn[j] == params[j] or paramsn[j] == paramsm[j]:
                paramsn[j] = clip(random.normal(params[j], sd[j]), inferior[j], superior[j])

            #print params[j], paramsn[j], paramsm[j]
            if j < 0: paramsn[:] = params[:]
            like1 = like_params(paramsn)            
        else:
            paramsn[:] = clip(random.normal(params, sd * 1.0/(-like0-100.0)), inferior, superior)
            like1 = like_params(paramsn)

        #print mascaras.shape, int(paramsn[0]),int(paramsn[1])
        if int(paramsn[0]) == 0: like1 -= 4
        if int(paramsn[1]) == 0: like1 -= 4
        r = log(random.random())
        dlike = like1 - like0

        if verbose:
            #print 'l0'like0,'l1',like1,'r',r,exp(r),'dlike',1.0/(-like0-100.0)
            print('traza iteracion %3d variable %3d'%(iteraciones,j), 'l0 %10.6f'%like0, 'l1 %10.6f'%like1, end=' ')
            print(params)
            print('%62s'%' ', end=' ')
            print(paramsm)
            print('%62s'%' ', end=' ')
            print(paramsn)
            print('%62s'%' ', end=' ')
            print(acepta)
            print('%62s'%' ', end=' ')
            print(sd)
            sys.stdout.flush()

        #print superior - inferior
            
        if dlike > r:
            like0 = like1
            if swt:
                acepta[j] = acepta[j] * 0.99 + 0.01
                paramsm[:] = paramsn[:]
        else:
            if swt:
                acepta[j] = acepta[j] * 0.99
        if swt:
            sd[j] *= (acepta[j] + 0.75)

        #print (acepta[j] + 0.75)
        #print (superior-inferior)/4.0
        sd = clip(sd,1e-9,(superior-inferior)/4.0)
                

        guardalog(like0, paramsm, paramsn, sd, acepta, like1, 0, 0, nproceso = nproceso[1])
        if j == 11: j = -1
                    

            
if __name__ == "__main__":
    nproceso = sys.argv[-2:]
    

    #ppp = [70.69137257007466,160.73797425871138,301.95588049130595,20.58093749684633,-8.876207447404777,0.4158576102391709,0.31958811138358134,-1.0,0.08295375085113783,0.0,1.0,0.01623871280660594,-101.396407376]
    #ppp = [11.065158905975721,168.88404748974676,303.95444512324684,26.478385940072855,-8.694048258965623,0.38563641429132245,1.0,-1.0,0.06814620669116134,0.0,1.0,0.0019999808855912043,-105.852960466]
    ppp = [42.30879513917583,156.83955954232553,301.9168766325274,16.1268516240829,-8.935665000082569,0.41478796260868456,0.012640945725831597,12.439085887209702,0.07803535155806585,0.0,1.0,0.002436000447251831,-101.533407245]
    #ppp = [58.35339080328758,163.84307710692158,302.5100951854642,16.733460171061562,-9.164084295712733,0.4166120781873127,0.34268290435826704,-1.0,0.07544326357266834,0.0,1.0,0.0,-102.339254324]
    priors = {"x0":   [[ppp[0]],  [3.4103040311747804e-05]],
              "y0":   [[ppp[1]],   [1.4223e-05]],
              "t0":   [[ppp[2]],   [0.028]],
              "radio": [[ppp[3]],  [0.00065]],
              "r":     [[ppp[8]], [3.394]],
              "oo":    [[ppp[4]],  [0.09]],
              "pe":    [[ppp[5]], [882.265]],
              "Kb":    [[ppp[6]],[0.044]],
              #"Kb" :   ppp[6],
              "Db":    [[ppp[7]], [0.005723936105603429]],
              #"Db":    [[ppp[3]], [0.005723936105603429]],
              #"Db":    ppp[7],
              "v" :    [[ppp[11]], [1.0]],
              #"v" :    [[0.0], [1.0]],
              #"v":     ppp[11],
              'per':   [[ppp[10]], [1.0]],
              'maxr':  [[ppp[9]], [1.0]],
    }

    for i in range(1000):
        gradiente_a(priors, nproceso)
        nproceso = nproceso[1], nproceso[1]

    
