# -*- coding: utf-8 -*-
import pyopencl as cl
from numpy import *
import sys, time

from pymc import binomial_like

import pylab


class gpu_base:
    """
    Class that performs matching and basis pursuit on space--time cubes
    """
    def __init__(self, dispositivo = 'gpu'):
        """
        Initialize the class, compile the opencl code, upload data, and initialize the params
        """
        #number of local threads in the opencl code (AMD=256, NVIDIA=1024)
        #Warning: using more than 256 threads might overflow the local memory
        self.matriz_concordia = load("matriz_concordia_x.npy").astype(float32)
        self.mascara_concordia = load("mascaras_x.npy").astype(int32)
        self.clima =     array(load("clima.npy").astype(float32)[:,1], copy = 1)
        self.mosquitos = load("datos_mosquitos_x.npy").astype(int32)
        self.init_cl(CPU = (dispositivo == 'cpu'))
        self.upload_data()

    def init_cl(self, CPU = False, device = 1, source_file = "gpu_mosquitos_gl.c"):
        """
        Compile the OpenCL code
        """
        
        platform = cl.get_platforms()
        
        #select device type
        if CPU:
            dvtype = cl.device_type.CPU
            print('Utilizando la CPU', end=' ')
            device = 0
            print(platform[device].get_devices(device_type = dvtype))
        else:
            dvtype = cl.device_type.GPU
            device = 0

        #initialize context
        self.ctx = cl.Context(devices = platform[device].get_devices(device_type = dvtype))

        #read the cl code from an file
        clcode =  open(source_file).read()
        #compiling the cl code
        self.prg = cl.Program(self.ctx, clcode).build(options = "", cache_dir=False)
        #keeping the CL variables into the object
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

    def upload_data(self):

        self.U  = zeros_like(self.matriz_concordia)
        self.MC = array(self.mascara_concordia, copy = 1)
        self.K  = float32((self.mascara_concordia > 0) +
                          0.1 * (self.mascara_concordia == 0) +
                          0.0001 * (self.mascara_concordia < 0))
        self.esperados = zeros((self.mosquitos.shape[0],), float32)

        self.params = zeros((10,), float32)
        #matrix with data
        self.cl_U         = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.U)
        self.cl_MC        = cl.Buffer(self.ctx, self.mf.READ_ONLY   | self.mf.COPY_HOST_PTR, hostbuf = self.MC)
        self.cl_K         = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.K)
        self.cl_Esp       = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.esperados)
        self.cl_mosquitos = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.mosquitos)
        self.cl_Clima     = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.clima)
        self.cl_params    = cl.Buffer(self.ctx, self.mf.READ_WRITE  | self.mf.COPY_HOST_PTR, hostbuf = self.params)


    def vermapa(self):
        import pylab
        print(self.MC.dtype, self.U.dtype, self.MC.max(), end=' ')
        #cl.enqueue_copy(self.queue, self.MC, self.cl_MC)
        #pylab.imshow(self.MC);pylab.show()
        print(self.MC.max(), end=' ')
        cl.enqueue_copy(self.queue, self.K, self.cl_K)
        #pylab.imshow(self.K);pylab.show()
        print(self.K.max(), self.K.mean(), end=' ')
        cl.enqueue_copy(self.queue, self.U, self.cl_U)
        pylab.imshow(self.U);pylab.show()
        print(self.U.max())
        
    def inicializa(self, Kbaldio):
        '''
        inicializacion del mapa
        se setean las variables de capacidad de carga y difusion
        el resto se inicia a cero
        
        self.U  = cl_array.to_device(queue, zeros_like(self.matriz_concordia))
        self.Uc = cl_array.zeros_like(self.U)
        self.MC = cl_array.to_device(queue, zeros_like(self.mascara_concordia))
        self.K  = cl_array.to_device(queue, float32((self.mascara_concordia > 0) + 0.1 * (self.mascara_concordia == 0) + 0.0001 * (self.mascara_concordia < 0)))

        '''
        n0 = (1 + array(self.U.shape) // 16) * 16

        self.prg.init_mapa(
            self.queue,
            tuple(n0),
            (16, 16),
            int32(self.U.shape[0]),
            int32(self.U.shape[1]),
            float32(Kbaldio),
            self.cl_K,
            self.cl_MC,
            self.cl_U)


    def simula(self, 
               fdia = 256,
               D = 16.0 ** 2 * pi,
               r = 1.5,
               x0 = 40,
               y0 = 160,
               oo = -10,
               pe = 0.5,
               dinvasion = 300.0,
               Kbaldio = 0.1,
               Dbaldio = 12. ** 2 * pi,
               per = 0.0,
               mxr = 1.0,
               vparms = False,
               hilosgpu = 256,
               v = 0.0025,
               vermapa = False):


        frames = fdia * self.clima.shape[0]
        #from scipy.special import logit, expit
        set_printoptions(precision = 4, linewidth = 240)
        n0 = (1 + array(self.U.shape) // 16) * 16

        #print x0, y0, dinvasion, D, sqrt(D / pi), oo, pe, Kbaldio, Dbaldio, r, per, mxr, v,
        self.params[:] = (oo, pe, mxr, per, r, D, Dbaldio, 25.0, 1.0/fdia, v)
        self.inicializa(Kbaldio)
        cl.enqueue_copy(self.queue, self.cl_params, self.params)

        tiempos = zeros((3,))

        if vermapa:
            #pylab.ion()
            pylab.imshow(self.U)
        
        for dia in range(self.clima.shape[0]):
            t0 = time.time()
            if dia >= floor(dinvasion):
                if dia == floor(dinvasion):
                    self.prg.siembra(
                        self.queue,
                        tuple(n0),
                        (16, 16),
                        int32(self.U.shape[0]),
                        int32(self.U.shape[1]),
                        float32(y0),
                        float32(x0),
                        float32(1.0 * (1 - (dinvasion - floor(dinvasion)))),
                        self.cl_U)

                elif dia == ceil(dinvasion):
                    self.prg.siembra(
                        self.queue,
                        tuple(n0),
                        (16, 16),
                        int32(self.U.shape[0]),
                        int32(self.U.shape[1]),
                        float32(y0),
                        float32(x0),
                        float32(1.0 * (dinvasion - floor(dinvasion))),
                        self.cl_U)
                
                t1 = time.time()
                #print tuple(n0), dia
                for hora in range(fdia):
                    self.prg.rdmosquitos(
                        self.queue,
                        tuple(n0),
                        (16, 16),
                        int32(self.U.shape[0]),
                        int32(self.U.shape[1]),
                        int32(dia),
                        self.cl_params,
                        self.cl_Clima,
                        self.cl_U,
                        self.cl_K,
                        self.cl_MC)
                if vermapa:
                    self.vermapa()
            else:
                t1 = t0
            t2 = time.time()

            #print self.mosquitos.shape
            self.prg.get_esperados(
                self.queue,
                (self.U.shape[0], self.U.shape[1], hilosgpu),
                (1, 1, hilosgpu),
                int32(self.U.shape[0]),
                int32(self.U.shape[1]),
                int32(dia),
                self.cl_U,
                self.cl_mosquitos,
                self.cl_Esp,
                uint32(self.mosquitos.shape[0]))
            
            t3 = time.time()
            tiempos += t1 - t0, t2 - t1, t3 - t2

        cl.enqueue_copy(self.queue, self.esperados, self.cl_Esp)
        return self.esperados

    def like(self,
             fdia = 256,
             D  = 14.0 ** 2 * pi,
             r  = 0.078,
             x0 = 29.7,
             y0 = 91.7,
             oo = -9.0,
             pe = 0.414,
             dinvasion =  296.4,
             Kbaldio = 0.0029,
             Dbaldio =  14.0 ** 2 * pi ,
             per = 0.0,
             mxr = 1.0,
             vparams = False,
             hilosgpu = 32,
             v = 0.0024,
             vermapa = False):
        #print fdia, D, r, x0, y0, oo, pe, dinvasion, Kbaldio, Dbaldio, per, mxr, hilosgpu, v,
        self.simula(fdia, D, r, x0, y0, oo, pe, dinvasion, Kbaldio, Dbaldio, per, mxr, hilosgpu = hilosgpu, v = v, vermapa = vermapa)
        l0 = binomial_like(self.mosquitos[:,4], 1., clip(self.esperados, 1e-6, 1-1e-6))
        if   isnan(l0): l0 = -1e9
        elif isinf(l0): l0 = -1e9
        self.likelihood = l0
        return l0

    def bic(self, encendido):
        self.deviance = -2 * self.likelihood
        self.BIC = self.deviance + encendido.sum() * log(self.mosquitos.shape[0])
        #print self.deviance, encendido.sum(), self.mosquitos.shape[0], self.BIC
        return self.BIC


def animacion():
    from matplotlib import pyplot as plt
    from matplotlib import animation
    
    from time import sleep

    fig = plt.figure()

    ax = plt.axes()

    gpmosquitos = gpu_base()
    
    a  = array(gpmosquitos.K, copy = 1)


    def haceplot():
        #
        spt = plt.subplot(2, 1, 1, title = 'Concordia')
        im = plt.imshow(a, interpolation='none', vmax = -8000, vmin = -8800, cmap = plt.cm.rainbow)
        ca = plt.colorbar()
        plt.grid()
        return im

    def init():
        im.set_data((a))
    
    def animate():
        a = im.get_array()
        im.set_array(a)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nnn, interval=1, blit=True)

    plt.show()
    logs.close()


    
if __name__ == '__main__':
    print("simulando")
    gpmosquitos = gpu_base()
    import time
    like0 = -1e9
    d11 = 0
    d01 = 0
    j0 = 0
    #pylab.ion()
    #like1 = gpmosquitos.like(hilosgpu = 32, y0 = 0, vermapa = True)
    #pylab.ioff()
    #pylab.show()
    animacion()

    
    for j in linspace(0, 257, 258):
        for i in range(1):
            print('y0',j, i, end=' ')
            t0 = time.time()
            like1 = gpmosquitos.like(hilosgpu = 32, y0 = j)
            print(like1, end=' ')
            print(time.time() - t0, end=' ')
            sys.stdout.flush()
            d11 = d01
            d01 = (like1 - like0) / (j - j0)
            if like1 > like0:
                print('*', end=' ')
                like0 = like1
            d2 = (d01 - d11)
            print(d11, d01, d2, d2**2)
            j0 = j
