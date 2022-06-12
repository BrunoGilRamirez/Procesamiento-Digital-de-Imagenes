from __future__ import print_function
from PIL import Image, ImageEnhance
import numpy as np
import math, random
import skimage.util as ski
import cv2
import matplotlib.pyplot as plt
import scipy.fftpack as fftim
alpha = 0.3
beta = 80
imge = 0
img2 =0
gamma_img=0
gamma=0
#------------------------------------------printers----------------------------------------------------------------
def print_img(nombre_imagen):
        pic=cv2.imread(nombre_imagen)
        pic=cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        plt.imshow(pic)#se imprime la imagen. 
        plt.show

def print_img_gray(nombre_imagen):
        pic=cv2.imread(nombre_imagen)
        plt.imshow(pic,cmap='gray')#se imprime la imagen. 
        plt.show()

def show_img(nombre_imagen):
        pic=cv2.imread(nombre_imagen)
        cv2.imshow("La imagen",pic)#se imprime la imagen. 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def evaluate(nombre_imagen,dir='',r=0):
        if type(nombre_imagen)==str:
                image=cv2.imread((dir+nombre_imagen),r)
        elif type(nombre_imagen)==np.ndarray:
                image=nombre_imagen
        return image
#----------------------------------------metodos para escalas de grises------------------------------------------------------------

def RBG2Gray_opencv  (nombre_imagen,dir=''):
         
         img=evaluate(nombre_imagen,dir)
         n=(dir+"gray-opencv_"+nombre_imagen)
         cv2.imwrite(n,img)
         plt.imshow(img, cmap="gray")
         return n

def RBG2Gray_EP(nombre_imagen,dir=''):
    
    img=evaluate(nombre_imagen,dir,r=1)
    i=0
    n=(dir+"gray-EP_"+nombre_imagen)
    while i<img.shape[1]:
        j=0
        while j < (img.shape[0]):
                blue=img[j,i,0]
                green=img[j,i,1]
                red=img[j,i,2]
                gray=(int(blue)+int(green)+int(red))/3
                g=int(gray)
                img[j,i,0]=g
                img[j,i,1]=g
                img[j,i,2]=g
                j+=1
        i+=1
    cv2.imwrite(n,img)
    return n,img

def RBG2Gray_HDTV (nombre_imagen,dir=''):
        foto = Image.open((dir+nombre_imagen))
        datos = foto.getdata()
        #para el calculo del promedio se utilizara la division entera con el operador de division doble "//" para evitar decimales
        promedio = [(((0.2126)*datos[x][0]) + ((0.7152)*datos[x][1]) + ((0.0722)*datos[x][2])) for x in range(len(datos))]
        imagen_gris = Image.new('L', foto.size)
        imagen_gris.putdata(promedio)
        name= dir+'gris_'+nombre_imagen
        imagen_gris.save(name)
        return name

def RBG2Gray_formatos_antiguos (nombre_imagen,dir=''):
        foto = Image.open((dir+nombre_imagen))
        datos = foto.getdata()
        #para el calculo del promedio se utilizara la division entera con el operador de division doble "//" para evitar decimales
        promedio = [(((0.299)*datos[x][0]) + ((0.587)*datos[x][1]) + ((0.114)*datos[x][2])) for x in range(len(datos))]
        imagen_gris = Image.new('L', foto.size)
        imagen_gris.putdata(promedio)
        name= dir+'gris_'+nombre_imagen
        imagen_gris.save(name)
        return name
    
def RBG2Gray_HDR (nombre_imagen):
        foto = Image.open(nombre_imagen)
        datos = foto.getdata()
        #para el calculo del promedio se utilizara la division entera con el operador de division doble "//" para evitar decimales
        promedio = [(((.2627)*datos[x][0]) + ((0.6780)*datos[x][1]) + ((0.0593)*datos[x][2])) for x in range(len(datos))]
        imagen_gris = Image.new('L', foto.size)
        imagen_gris.putdata(promedio)
        name= 'gris_'+nombre_imagen
        imagen_gris.save(name)
        return name
#----------------------------------------------Funciones de contrastes-------------------------------------------------------------
def pot_contrast(nombre_imagen,c,Gamma,dir='',ban=True):
        img=evaluate(nombre_imagen,dir)
        plt.title("Imagen antes de la transformacion")
        plt.imshow(img,cmap="gray")
        plt.show()
        hist(img)
        pot= np.zeros(img.shape, img.dtype)
        g= 1.0 / Gamma
        pot= c*np.power(img,g)
        max=np.max(pot)
        img_pot=np.uint8(pot/max*255)
        histo=hist(img_pot)
        if(ban==True):
                plt.title("Transformacion Gamma(potencia)")
                plt.imshow(img_pot,cmap="gray")
                plt.show() 
        else:
                return img_pot

def log_contrast (nombre_imagen,c,dir='',ban=True): 
        img= evaluate(nombre_imagen,dir)
        plt.title("Imagen antes de la transformacion")
        plt.imshow(img,cmap="gray")
        plt.show()
        hist(img)
        log= np.zeros(img.shape, img.dtype)
        log= c*np.log(1+img)
        max=np.max(log)
        img_log=np.uint8(log/max*255)
        histo=hist(img_log)
        if(ban==True):
                plt.title("Transformacion logaritmica")
                plt.imshow(img_log,cmap="gray")
                plt.show()
        else:
                return img_log

def contrastBy_histogram(nombre_imagen,dir='',scale=1,ban=True):
        img = evaluate(nombre_imagen,dir)
        plt.title("im")
        plt.imshow(img,cmap="gray")
        plt.show()
        cdf=hist(img)
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (scale*((cdf_m - cdf_m.min())))/(cdf_m.max()-cdf_m.min())*255
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]
        if(ban==True):
                plt.title("im2")
                plt.imshow(img2,cmap="gray")
                plt.show()
                hist(img2)
                n=(dir+"contras_byhist_"+nombre_imagen) 
                cv2.imwrite(n,img2)
        return img2,n

def equalitation_by_hist(nombre_imagen,dir='',scale=1):
        img=evaluate(nombre_imagen,dir)
        cdf=Histograma(img)
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (scale*((cdf_m - cdf_m.min())))/(cdf_m.max()-cdf_m.min())*255
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]
        return img2

def Histograma(imagen,dir=''):
        img=evaluate(imagen,dir)
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        return cdf

#----------------------------------------------------- Padding ---------------------------------------------------------------------------
def padding_CVcomparison(nombre_imagen,dir=''):
        name,img=RBG2Gray_EP(nombre_imagen,dir)#escala de grises con la escala promediada
        pic=cv2.imread((dir+nombre_imagen),0)#escala de grises con la funcion imread
        img_pad = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0,0,0))
        fig = plt.figure(figsize=(10, 10))# crear figura
        # establecer los valores del tamaño de los subplots
        rows = 2
        columns = 2
        # incluir el subplot en la primera posicion
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        plt.title("con Escala promediada")
        # incluir el subplot en la segunda posicion
        fig.add_subplot(rows, columns, 2)
        # mostrando la imagen
        plt.imshow(pic,cmap="gray")
        plt.title("con OpenCV (imread)")
        fig.add_subplot(rows, columns, 3)
        # mostrando la imagen
        plt.imshow(img_pad,cmap="gray")
        plt.title("zero-padding con openCV")
        plt.show()
#----------------------------------------------------convolucion-------------------------------------------------------------------------------
def miConvo(nombre_imagen,dir=''):
        m1=evaluate(nombre_imagen,dir)
        nf,nc=m1.shape #se obtiene las filas y columna de la imagen
        h=np.full((3,3),(1/9)) #se inicializa un np.array que tenga 1/9
        m2=np.zeros((nf,nc)) #se inicializa un np.array del mismo tamaño de la imagen
        #los bucles siguiente se realiza una convolucion donde se hace directamente pixel por pixel
        for f in range(1,nf-1): #filas
                for c in range(1,nc-1): #columnas
                        sum=0 #se inicializa sum
                        for ff in range(-1,1): #para recorrer las filas de la mascara o kernel
                                for cc in range(-1,1): #para recorrer las columnas de  la mascara o kernel
                                        x,y,x1,y1=(f+ff+1),(c+cc+1),(ff+2),(cc+2) #se calculan los indices
                                        sum+=(m1[x,y]*h[x1,y1]) #se hace la operacion
                        m2[f,c]=sum #se reasignan las columnnas 
        #ahora se hace una convolucion con filter2D, de la imagen original (src= m1)
        m3=cv2.filter2D(m1,-1,h)#la profundidad de -1, o sea igual que la de src, y el mismo kernel usado anteriormente (h)
        fig = plt.figure(figsize=(10, 10))# crear figura
        # establecer los valores del tamaño de los subplots
        rows = 2
        columns = 2
        # incluir el subplot en la primera posicion
        fig.add_subplot(rows, columns, 1)
        plt.imshow(m1,cmap="gray")
        plt.title("Imagen original")
        # incluir el subplot en la segunda posicion
        fig.add_subplot(rows, columns, 2)
        # mostrando la imagen
        plt.imshow(m2,cmap="gray")
        plt.title("Imagen convolucionada")
        # incluir el subplot en la cuarta posicion
        fig.add_subplot(rows, columns, 4)
        # mostrando la imagen
        plt.imshow(m3,cmap="gray")
        plt.title("Imagen convolucionada con cv2.filter2D")
        plt.show()

def filt(m1,h):
        nf,nc=m1.shape #se obtiene las filas y columna de la imagen
        m2=np.zeros((nf,nc)) #se inicializa un np.array del mismo tamaño de la imagen
        #los bucles siguiente se realiza una convolucion donde se hace directamente pixel por pixel
        for f in range(1,nf-1): #filas
                for c in range(1,nc-1): #columnas
                        sum=0 #se inicializa sum
                        for ff in range(-1,1): #para recorrer las filas de la mascara o kernel
                                for cc in range(-1,1): #para recorrer las columnas de  la mascara o kernel
                                        x,y,x1,y1=(f+ff+1),(c+cc+1),(ff+2),(cc+2) #se calculan los indices
                                        sum+=(m1[x,y]*h[x1,y1]) #se hace la operacion
                        m2[f,c]=sum #se reasignan las columnnas 
        #ahora se hace una convolucion con filter2D, de la imagen original (src= m1)
        
        fig = plt.figure(figsize=(10, 10))# crear figura
        # establecer los valores del tamaño de los subplots
        rows = 2
        columns = 2
        # incluir el subplot en la primera posicion
        fig.add_subplot(rows, columns, 1)
        plt.imshow(m1,cmap="gray")
        plt.title("Imagen original")
        # incluir el subplot en la segunda posicion
        fig.add_subplot(rows, columns, 2)
        # mostrando la imagen
        plt.imshow(m2,cmap="gray")
        plt.title("Imagen Filtrada")
        # incluir el subplot en la cuarta posicion
        plt.show()
#------------------------------------------------------unsharpen filter------------------------------------------------
def unsharpen(nombre_imagen,dir='',intensity=.9899996,sigma=15,ban=True):
        '''
        funcion que afila la imagen de la luna ejercicio 6 con las operaciones aritmeticas de una imagen.
        '''
        image=cv2.imread((dir+nombre_imagen),0)
        immf =  cv2.medianBlur(image,sigma)
        kr1=image-intensity*immf
        imf=image+kr1
        imf=cv2.convertScaleAbs(imf)
        if(ban==True):
                fig = plt.figure(figsize=(15, 30));rows,columns = 3,2# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 1),plt.imshow(image,cmap="gray"),plt.title("imagen original")
                fig.add_subplot(rows, columns,3),plt.imshow(immf,cmap="gray"),plt.title(" median_filter")
                fig.add_subplot(rows, columns, 2),plt.imshow(imf,cmap="gray"),plt.title("Unsharpened")
                fig.add_subplot(rows, columns, 4),plt.imshow(kr1,cmap="gray"),plt.title("Kernel escalado")
                plt.show()
        else:
                return imf

def unsharpen_lp(nombre_imagen,dir='',intensity=.9899996,sigma=15,ban=True):
        image=evaluate(nombre_imagen,dir)
        immf =  cv2.medianBlur(image,sigma)
        lap = cv2.Laplacian(image,cv2.CV_16S)
        lp= cv2.convertScaleAbs(lap)
        sharp = image - intensity*lap
        sharp =cv2.convertScaleAbs(sharp)
        if(ban==True):
                fig = plt.figure(figsize=(15, 30));rows,columns = 3,2# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 1),plt.imshow(image,cmap="gray"),plt.title("imagen original")
                fig.add_subplot(rows, columns,5),plt.imshow(immf,cmap="gray"),plt.title(" median_filter")
                fig.add_subplot(rows, columns, 3),plt.imshow(lap,cmap="gray"),plt.title("Laplaciano en 64F")
                fig.add_subplot(rows, columns, 4),plt.imshow(lp,cmap="gray"),plt.title("Laplaciano en 8uint")
                fig.add_subplot(rows, columns, 2),plt.imshow(sharp,cmap="gray"),plt.title("Unsharpened")
                plt.show()
        else:
                return sharp

def unsharpen_lp_frec(nombre_imagen,dir='',intensity=.9899996,sigma=15,ban=True):
        image=evaluate(nombre_imagen,dir)
        immf =  cv2.medianBlur(image,sigma)
        lap,con = freclaplacian_filter(image)
        lp= cv2.convertScaleAbs(lap)
        sharp = image + intensity*lap
        sharp =cv2.convertScaleAbs(sharp)
        if(ban==True):
                fig = plt.figure(figsize=(15, 15));rows,columns = 2,2# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 1),plt.imshow(image,cmap="gray"),plt.title("imagen original")
                fig.add_subplot(rows, columns, 2),plt.imshow(sharp,cmap="gray"),plt.title("Unsharpened Image")
                fig.add_subplot(rows, columns, 3),plt.imshow(lp,cmap="gray"),plt.title("Laplaciano En el dominio de la frecuencia")
                fig.add_subplot(rows, columns,4),plt.imshow(immf,cmap="gray"),plt.title(" median_filter")
                plt.show()
        else:
                return sharp
#------------------------------------------------------------Foco_luminoso y circulo------------------------------------------------------------
def foco_luminoso(n=300,m=300):
        a,b=150,150;
        I=np.zeros((m, n))
        for x in range(1,m):
                for y in range(1,n):
                        I[x,y]=(255-((x-a)**2+(y-b)**2)**(.5))/255

        return I;

def circulo(n=100,m=100):
        I= np.ones((m, n))
        for x in range(1,m):
                for y in range(1,n):
                        if ((x-50)**2+(y-50)**2)<700:
                                I[x,y] =0
        return I;

#------------------------------------------------------------- filtros Ideales ----------------------------------------------------------------
def ideal_lw_filters(imagen,d_0 = 30.0,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H=np.ones((M,N))
        center1 = M/2
        center2 = N/2
        # defining the convolution function for ILPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to eliminate
                        # high frequency
                        if r > d_0:
                                H[i,j]=0.0
        # converting H to an image
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f= abs(fftim.ifft2(con))
        return f,con

def ideal_butterworth(imagen,d_0 = 30.0,t1 = 1,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H = np.ones((M,N))
        center1 = M/2
        center2 = N/2
        t2 = 2*t1
        # defining the convolution function for BLPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to
                        # eliminate high frequency
                        if r > d_0:
                                H[i,j] = 1/(1 + (r/d_0)**t2)
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        return f,con

def ideal_hgh_filt(imagen,d_0 = 30.0,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H = np.ones((M,N))
        center1 = M/2
        center2 = N/2
        # defining the convolution function for IHPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to
                        # eliminate low frequency
                        if 0 < r < d_0:
                                H[i,j] = 0.0
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        return f,con

def gauss_hg_filt(imagen,d_0 = 30.0,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and values in H are initialized to 1
        H = np.ones((M,N))
        center1 = M/2
        center2 = N/2
        t1 = 2*d_0
        # defining the convolution function for GHPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to
                        # eliminate low frequency
                        if 0 < r < d_0:
                                H[i,j] = 1 - math.exp(-r**2/t1**2)
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        return f,con

def bandass_filt(imagen,d_0 = 30.0,d_1 = 50.0,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H = np.zeros((M,N))
        center1 = M/2
        center2 = N/2
        # defining the convolution function for bandpass
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using min and max cut-off to create
                        # the band or annulus
                        if r > d_0 and r < d_1:
                                H[i,j] = 1.0
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        return f,con

def freclaplacian_filter(imagen,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        #generate with the Laplacian Frequency domain transformacion the kernel H 
        u, v = np.mgrid[-1:1:2.0/d.shape[0], -1:1:2.0/d.shape[1]]
        D = np.sqrt(u**2 + v**2)
        H = -4 * np.pi**2 * D**2
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        mask2 = np.ones(d.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = f * mask2  # g(x,y) * (-1)^(x+y)
        # (8) Intercept the upper left corner, the size is equal to the input image
        result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
        imgFilter = result.astype(np.uint64)
        row, col = b.shape[:2]  # The height and width of the picture
        imgFilter = imgFilter[:row, :col]
        return imgFilter,con 

def ideal_bttrworth_hg(imagen,d_0 = 30.0,t1 = 1,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H = np.ones((M,N))
        center1 = M/2
        center2 = N/2
        t1 = 1 # the order of BHPF
        t2 = 2*t1
        # defining the convolution function for BHPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to
                        # eliminate low frequency
                        if 0 < r < d_0:
                                H[i,j] = 1/(1 + (r/d_0)**t2)
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        return f,con

def gauss_lw_filt(imagen,d_0 = 30.0,dir=''):
        b=evaluate(imagen,dir)
        # performing FFT
        c = fftim.fft2(b)
        # shifting the Fourier frequency image
        d = fftim.fftshift(c)
        # intializing variables for convolution function
        M = d.shape[0]
        N = d.shape[1]
        # H is defined and
        # values in H are initialized to 1
        H = np.ones((M,N))
        center1 = M/2
        center2 = N/2
        t1 = 2*d_0
        # defining the convolution function for GLPF
        for i in range(1,M):
                for j in range(1,N):
                        r1 = (i-center1)**2+(j-center2)**2
                        # euclidean distance from
                        # origin is computed
                        r = math.sqrt(r1)
                        # using cut-off radius to
                        # eliminate high frequency
                        if r > d_0:
                                H[i,j] = math.exp(-r**2/t1**2)
        # performing the convolution
        con = d * H
        # computing the magnitude of the inverse FFT
        f = abs(fftim.ifft2(con))
        # e is converted from an ndarray to an imag
        return f,con

def Put_3d(f,con,x=135,y=135,dx=154,dy=154):
        xs,ys,zs = abs(con[:,0]),abs(con[:,1]),abs(con[:,2])
        fig = plt.figure(figsize=(15, 15));rows,columns = 2,2# crear figura, establecer los valores del tamaño de los subplots
        fig.add_subplot(rows, columns, 1),plt.imshow(f,cmap="gray"),plt.title("Original")
        fig.add_subplot(rows, columns, 2),plt.imshow(abs(con[x:dx,y:dy])),plt.title("Transformacion de Fourier en 2D")
        ax=fig.add_subplot(rows, columns, 3,projection='3d').plot_trisurf(xs-xs.mean(), ys-ys.mean(), zs, linewidth=0)
        fig.colorbar(ax),fig.tight_layout()
        plt.show()

def Put_only2D(f,con,x=135,y=135,dx=154,dy=154):
        xs,ys,zs = abs(con[:,0]),abs(con[:,1]),abs(con[:,2])
        fig = plt.figure(figsize=(15, 30));rows,columns = 1,2# crear figura, establecer los valores del tamaño de los subplots
        fig.add_subplot(rows, columns, 1),plt.imshow(f,cmap="gray"),plt.title("Original")
        fig.add_subplot(rows, columns, 2),plt.imshow(abs(con[x:dx,y:dy])),plt.title("Transformacion de Fourier en 2D")
        plt.show()

def sub2print(f,con,cc,a='',b='',c=''):
        fig = plt.figure(figsize=(45, 15));rows,columns = 1,3# crear figura, establecer los valores del tamaño de los subplots
        fig.add_subplot(rows, columns, 1),plt.imshow(f,cmap="gray"),plt.title(a)
        fig.add_subplot(rows, columns, 2),plt.imshow(con,cmap="gray"),plt.title(b)
        fig.add_subplot(rows, columns, 3),plt.imshow(cc,cmap="gray"),plt.title(c)
        plt.show()

#----------------------------------------------------- Sobel Operator ----------------------------------------------------------------
def Sobel_operator(nombre_imagen,dir='',ban=True):
        '''
        funcion que utilizara el operador sobel para resaltar defectos en un lente.
        '''
        image=evaluate(nombre_imagen,dir)
        sobelxy=Sob(image) 
        if(ban==True):
                fig = plt.figure(figsize=(30, 25));rows,columns = 2,1# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 1),plt.imshow(image,cmap="gray"),plt.title("imagen original")
                fig.add_subplot(rows, columns,2),plt.imshow(sobelxy,cmap="gray"),plt.title("Filtro Sobel en X y Y")
                plt.show()
        else:
                return sobelxy
def Sob(image):
        x = cv2.Sobel(image,3,1,0)  
        y = cv2.Sobel(image,2,0,1)  
        absX = cv2.convertScaleAbs(x)   # Transferencia de regreso a uint8  
        absY = cv2.convertScaleAbs(y)  
        sobelxy = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return sobelxy

#------------------------------------------------------ combinando metodos de realce espacial ----------------------------------------------------
def combinando(nombre_imagen,dir='',intensity=.9899996,intensit=.5,sigma=3,gamma=1):
        '''
        funcion que combina todos los metodos vistos en la practica para afilar las imagenes 
        y resaltar bordes horizontales y verticales. 
        '''
        image=cv2.imread((dir+nombre_imagen),0)
        lp= cv2.Laplacian(image, cv2.CV_16S)
        shpimg=cv2.convertScaleAbs((image-(intensity*lp)))
        sobel=cv2.convertScaleAbs(Sob(image))
        sobel_E=cv2.GaussianBlur(sobel,(5,5),sigma)
        fmask=shpimg*(intensit*sobel_E)
        img_g=cv2.convertScaleAbs(image+fmask)
        img_h=gamma_correction(img_g,gamma)
        fig = plt.figure(figsize=(16, 48));rows,columns = 4,2# crear figura, establecer los valores del tamaño de los subplots
        #agregar a los subplots las imagenes.
        fig.add_subplot(rows, columns, 1),plt.imshow(image,cmap="gray"),plt.title("A) imagen original")
        fig.add_subplot(rows, columns, 2),plt.imshow(lp,cmap="gray"),plt.title("B)Laplacian de A)")
        fig.add_subplot(rows, columns, 3),plt.imshow(shpimg,cmap="gray"),plt.title("sharpened de A+B")
        fig.add_subplot(rows, columns, 4),plt.imshow(sobel,cmap="gray"),plt.title("Sobel de A)")
        fig.add_subplot(rows, columns, 5),plt.imshow(sobel_E,cmap="gray"),plt.title("E) Sobel con GaussianBlur 5*5")
        fig.add_subplot(rows, columns, 6),plt.imshow(fmask,cmap="gray"),plt.title("B)Mascara formada por C*E")
        fig.add_subplot(rows, columns, 7),plt.imshow(img_g,cmap="gray"),plt.title("sharpened de A+F")
        fig.add_subplot(rows, columns, 8),plt.imshow(img_h,cmap="gray"),plt.title(f"TRANSFORMACION DE POTENCIA EN G)gamma={gamma}")
        plt.show()

#-------------------------------------------------- ruido impulsivo ----------------------------------------------------------------
def ruidoimp(nombre_imagen,dir='',p=1,imin=1,imax=1):
        a=cv2.imread((dir+nombre_imagen),0)
        fig = plt.figure(figsize=(10, 10))# crear figura
        rows,columns = 2,2
        # incluir el subplot en la primera posicion
        fig.add_subplot(rows, columns, 1)
        plt.imshow(a,cmap="gray"), plt.title("Imagen original")
        m,n= a.shape
        np= math.ceil((p/10)*m*n)
        turn=0
        b=a
        for i in range(1,np):
                x,y = (m*random.random()),(n*random.random())
                k = round(x)-1
                l = round(y)-1
                if turn==0:
                        turn=1
                        b[k,l]=imax
                else:
                        turn=0
                        b[k,l]=imin
        
        # incluir el subplot en la segunda posicion
        fig.add_subplot(rows, columns, 2)
        # mostrando la imagen
        plt.imshow(b,cmap="gray")
        plt.title("Imagen con ruido impulsivo")
        plt.show()

def threshold (nombre_imagen,dir='',ban=True):
        if type(nombre_imagen)==str:
                img=cv2.imread((dir+nombre_imagen),0)
        elif type(nombre_imagen)==np.ndarray:
                img=nombre_imagen
        blur = cv2.medianBlur(img,15)
        th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        if(ban==True):
                fig = plt.figure(figsize=(15, 10));rows,columns = 2,2# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 1),plt.imshow(img,cmap="gray"),plt.title("imagen original")
                fig.add_subplot(rows, columns, 3),plt.imshow(blur,cmap="gray"),plt.title("imagen con Blur")
                fig.add_subplot(rows, columns, 4),plt.imshow(th1,cmap="gray"),plt.title("thresholding")
                plt.show()
        
        return th1

#---------------------------------------------------brillo y gamma---------------------------------------------------------- 
def gamma_correction(nombre_imagen,g,c=.25,dir=''):
        img=evaluate(nombre_imagen,dir)
        pot= np.zeros(img.shape, img.dtype)
        pot= c*np.power(img,g)
        max=np.max(pot)
        img_pot=np.uint8(pot/max*255)
        return img_pot

#-------------------------------------------------------Rotaciones-----------------------------------------------------------------
def rotacion_lineal (imagen,teta,dir='',ban=True):
        img = evaluate(imagen,dir)
        ancho = img.shape[1] #columnas
        alto = img.shape[0] # filas
        Matrix = cv2.getRotationMatrix2D((ancho/2,alto/2),teta,1) # el formato es cv2.getRotationMatrix2D(centro, angulo, escala) 
        dst = cv2.warpAffine(img,Matrix,(ancho,alto),flags=cv2.INTER_LINEAR)
        if(ban==True):
                plt.imshow(dst), plt.xlabel("Lineal"),plt.show() 

def rotacion_nearestNeighbor (imagen,teta,dir='',ban=True):
        img = evaluate(imagen,dir)
        ancho = img.shape[1] #columnas
        alto = img.shape[0] # filas
        Matrix = cv2.getRotationMatrix2D((ancho/2,alto/2),teta,1) # el formato es cv2.getRotationMatrix2D(centro, angulo, escala) 
        dst = cv2.warpAffine(img,Matrix,(ancho,alto),flags=cv2.INTER_NEAREST)
        if(ban==True):
                plt.imshow(dst),plt.xlabel("Nearest Neighbor"),plt.show() 

def rotacion_cubic (imagen,teta,dir='',ban=True):
        img = evaluate(imagen,dir)
        ancho = img.shape[1] #columnas
        alto = img.shape[0] # filas
        Matrix = cv2.getRotationMatrix2D((ancho/2,alto/2),teta,1) # el formato es cv2.getRotationMatrix2D(centro, angulo, escala)
        dst = cv2.warpAffine(img,Matrix,(ancho,alto),flags=cv2.INTER_CUBIC)
        if(ban==True):
                plt.imshow(dst),plt.xlabel("bicubic"),plt.show() 

def rotacion_bilineal (imagen,teta,dir='',ban=True):
        img = evaluate(imagen,dir)
        ancho = img.shape[1] #columnas
        alto = img.shape[0] # filas
        Matrix = cv2.getRotationMatrix2D((ancho/2,alto/2),teta,1) # el formato es cv2.getRotationMatrix2D(centro, angulo, escala) 
        dst = cv2.warpAffine(img,Matrix,(ancho,alto))
        if(ban==True):
                plt.imshow(dst),plt.xlabel("Bilineal"),plt.show()  

#-------------------------------------------------------histogramas------------------------------------------------------------------
def histograma_colorfull(imagen,label):
        img = imagen 
        color = ('b', 'g', 'r') 
        for i, col in enumerate(color): 
                histr = cv2.calcHist([img], [i], None, [256], [0, 256]) 
                plt.plot(histr, color = col) 
                plt.xlim([0, 256]) 
        plt.title("Histograma de imagen "+label),plt.xlabel("Rango Dinamico"),plt.ylabel("# de Pixeles"),
        plt.show()

def hist(imagen,dir='',ban=True):
        img=evaluate(imagen,dir)
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        if(ban==True):
                cdf_normalized = cdf * float(hist.max()) / cdf.max()
                plt.plot(cdf_normalized, color = 'b')
                plt.hist(img.flatten(),256,[0,256], color = 'r')
                plt.xlim([0,256])
                plt.legend(('cdf','histograma'), loc = 'upper left')
                plt.show()
        return cdf

def histograma(imagen,dir='',ban=True):
        img=evaluate(imagen,dir)
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        if(ban==True):
                fig = plt.figure(figsize=(15, 34));rows,columns = 6,2# crear figura, establecer los valores del tamaño de los subplots
                #agregar a los subplots las imagenes.
                fig.add_subplot(rows, columns, 2), plt.plot(cdf_normalized, color = 'b'), plt.hist(img.flatten(),256,[0,256], color = 'r'),plt.xlim([0,256]),plt.legend(('cdf','histograma'), loc = 'upper left')
                fig.add_subplot(rows, columns, 1),plt.imshow(imagen,cmap="gray"),plt.title("imagen")
                plt.show()
#-------------------------------------------------deteccion de una imagen falsa------------------------------------------
def fake_detectition(nombre_imagen,nombre_imagen1):
        img = cv2.imread(nombre_imagen,0)
        plt.title("imagen de referencia") 
        plt.imshow(img,cmap="gray") 
        plt.show() 
        img2 = cv2.imread(nombre_imagen1,0)
        plt.title("imagen alterada") 
        plt.imshow(img2,cmap="gray") 
        plt.show() 
        ancho = img.shape[1] #columnas
        alto = img.shape[0] # filas
        # Traslación
        y=-3
        for x in range(-3,4):
                for y in range(-3,4):
                        m = np.float32([[1,0,x],[0,1,y]])
                        img1 = cv2.warpAffine(img,m,(ancho,alto))
                        res= cv2.subtract(img2,img1)
                        t=(f"traslacion (x,y)= ({x},{y}) imagen resultante")
                        plt.title(t) 
                        plt.imshow(res,cmap="gray") 
                        plt.show()
                        
#------------------------------------------------negativo de una imagen---------------------------------------------------------------------
def negative (nombre_imagen,dir=''):
        if type(nombre_imagen)==str:
                img=cv2.imread((dir+nombre_imagen),1)
        elif type(nombre_imagen)==np.ndarray:
                img=nombre_imagen 
        plt.title("Imagen Original")
        plt.imshow(img) 
        plt.show()  
        histograma_colorfull(img,"original")   
        img_neg = 1 - img 
        plt.title("Imagen Negativa") 
        plt.imshow(img_neg) 
        plt.show() 
        histograma_colorfull(img_neg,"negativa") 
        n=(dir+"negative_"+nombre_imagen) 
        cv2.imwrite(n,img_neg)
        return n

#------------------------------------------------auxiliares de funciones-------------------------------------------------------------

def adjust_brightness(nombre_imagen, brightness_factor):
    img = Image.open(nombre_imagen)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img 

def contrast_window(nombre_imagen):
        global alpha,beta, imge, img2
        alpha = 0.3
        beta = 80
        imge = cv2.imread(nombre_imagen)
        img2 = cv2.imread(nombre_imagen)
        cv2.namedWindow('image')
        cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
        cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
        cv2.setTrackbarPos('Alpha', 'image', 100)
        cv2.setTrackbarPos('Beta', 'image', 10)
        while (True):
                cv2.imshow('image', imge)
                if cv2.waitKey(10)==ord("q")or cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:
                        break
        cv2.destroyAllWindows()
        n=("contrast_"+nombre_imagen)
        cv2.imwrite(n,imge)
        return n

def gamma_adjustable (nombre_imagen1):
        global gamma, gamma_img, nombre_imagenn
        nombre_imagenn=nombre_imagen1
        cv2.namedWindow('image_Gamma')
        cv2.createTrackbar('gamma', 'image_Gamma', 1000, 2000, updateGamma)
        cv2.setTrackbarPos('gamma', 'image_Gamma', 1000)
        gamma_img=cv2.imread(nombre_imagenn)
        while (True):
                cv2.imshow('image_Gamma', gamma_img)
                if cv2.waitKey(10)==ord("q")or cv2.getWindowProperty('image_Gamma',cv2.WND_PROP_VISIBLE) < 1:
                        break
        cv2.destroyAllWindows()

def updateAlpha(x):
    global alpha, imge, img2
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    alpha = alpha * 0.01
    imge = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
def updateBeta(x):
    global beta, imge, img2
    beta = cv2.getTrackbarPos('Beta', 'image')
    imge = np.uint8(np.clip((alpha * img2 + beta), 0, 255))   

def updateGamma(x):
        global gamma, gamma_img, nombre_imagenn
        gamma_img=cv2.imread(nombre_imagenn)
        gamma = cv2.getTrackbarPos('gamma', 'image_Gamma')
        if gamma>0: 
                gamma=gamma*.001 
        elif gamma==0:
                gamma=.7
        print (gamma)
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        gamma_img=cv2.LUT(gamma_img, table)