from calendar import c
from tkinter import *
from turtle import width
import numpy as np
from pyparsing import col 
from pinn_code import train
from model import *
from plot import * 

#action des boutons du cadre de l'équation
def actionbuttonplus():
    valueequa.set(entreeequ.get() + ' + ')

def actionbuttonminus():
    valueequa.set(entreeequ.get() + ' - ')

def actionbuttonsin():
    valueequa.set(entreeequ.get() + 'sin (')

def actionbuttoncos():
    valueequa.set(entreeequ.get() + 'cos (')

def actionbuttonsquare():
    valueequa.set(entreeequ.get() + ') ^ 2')

def actionbuttonopen():
    valueequa.set(entreeequ.get() + '(')

def actionbuttonclose():
    valueequa.set(entreeequ.get() + ')')

def actionbuttonx():
    valueequa.set(entreeequ.get() + 'x')

def actionbuttony():
    if int(entreedim.get()) >= 2 :
        valueequa.set(entreeequ.get() + 'y')

def actionbuttonz():
    if int(entreedim.get()) == 3 :
        valueequa.set(entreeequ.get() + 'z')

def actionvalider():
    c,a,dimension,tmin,tmax,xmin,xmax,N_b,N_r,N_0,lr,epochs = int(entreec.get()), 0.5, int(entreedim.get()), 0., 1., 0., 1., int(entreeNb.get()), int(entreeNr.get()), int(entreeN0.get()), float(entreelr.get())*1e-5, int(entreeepochs.get())
    X_data,u_data,time_x,X_r = set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)
    #plot_training_points(dimension,time_x)
    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)
    plot1dgrid_real(lb,ub,200,lambda x: true_u(x,a,c),999999)
    opt = keras.optimizers.Adam(learning_rate=lr)
    hist = []
    pinn = PINN(dimension+1,1,dimension,ub,lb,c)
    pinn.compile(opt)
    train(epochs,pinn,X_r,X_data,u_data,true_u,N=100,dimension=dimension,batch_size=450)

    """
    model = pinn.model
    N = 70
    fps = 5
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1d(lb,ub,N,tspace,model,fps)
    N = 100
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1dgrid(lb,ub,N,model,0)
    """
    
    fenetre.quit()

## fenetre tkinter
fenetre = Tk()
fenetre.title('PINN')

top=Frame(fenetre)
bottom = Frame(fenetre)

cadreentree = Frame(top, border = 2)

valider=Button(bottom, text="Valider", command=actionvalider)
valider.pack()

#cadre paramètres
l = LabelFrame(cadreentree, text="paramètres", padx=20, pady=5)

#sous-cadre pour récupérer c
zonec = Frame(l)
labelc = Label(zonec, text="c :")
labelc.pack(side= LEFT)
valuec = StringVar() 
entreec = Entry(zonec, textvariable=valuec, width=5)
valuec.set('1')
entreec.pack(side = RIGHT, padx= 15)
zonec.pack(fill="both", expand="yes", side = LEFT)


#sous-cadre pour récupérer dim
zonedim = Frame(l)
labeldim = Label(zonedim, text="dimension :")
labeldim.pack(side = LEFT)
valuedim = StringVar()
entreedim = Spinbox(zonedim, textvariable=valuedim, from_=1, to=3, width = 5)
entreedim.pack(side = RIGHT)
labeldim.pack(side = LEFT)
zonedim.pack(fill="both", expand="yes", side = LEFT)

l.pack(fill="both", expand="yes", side= TOP)
Label(l).pack()

#cadre conditionsinit
equa = LabelFrame(cadreentree, text="conditions initiales", padx=20, pady=5)
valueequa = StringVar()
valueequa.set('')
entreeequ = Entry(equa, textvariable=valueequa, width = 30)
entreeequ.pack(side= TOP, padx = 10, pady = 10)

#bouton pour écrire
bout = Frame(equa)
up = Frame(bout)
down = Frame(bout)
buttonplus = Button(up, text = '+', command= actionbuttonplus, width = 3)
buttonplus.pack(side = LEFT)
buttonminus = Button(up, text = '-', command= actionbuttonminus, width = 3)
buttonminus.pack(side = LEFT)
buttonsin = Button(up, text = 'sin', command= actionbuttonsin, width = 3)
buttonsin.pack(side = LEFT)
buttoncos = Button(up, text = 'cos', command= actionbuttoncos, width = 3)
buttoncos.pack(side = LEFT)
buttonsquare = Button(up, text = '^2', command= actionbuttonsquare, width = 3)
buttonsquare.pack(side = LEFT)
buttonopen = Button(down, text = '(', command= actionbuttonopen, width = 3)
buttonopen.pack(side = LEFT)
buttonclose = Button(down, text = ')', command= actionbuttonclose, width = 3)
buttonclose.pack(side = LEFT)
buttonx = Button(down, text = 'x', command= actionbuttonx, width = 3)
buttonx.pack(side = LEFT)
buttony = Button(down, text = 'y', command= actionbuttony, width = 3)
buttony.pack(side = LEFT)
buttonz= Button(down, text = 'z', command= actionbuttonz, width = 3)
buttonz.pack(side = LEFT)
up.pack(side=TOP)
down.pack(side = BOTTOM)



bout.pack(side=BOTTOM)


equa.pack(side=BOTTOM)


cadreentree.pack(side = LEFT)

cadredroit = Frame(top, border = 2, width = 50)
l2 = LabelFrame(cadredroit, text="paramètres d'apprentissage", padx=20, pady=8)

#sous-cadre pour récupérer épochs
zoneepochs = Frame(l2)
labelepochs = Label(zoneepochs, text="Nombre d'épochs :")
labelepochs.pack(side= LEFT)
valueepochs = StringVar()
entreeepochs = Entry(zoneepochs, textvariable=valueepochs, width=5)
valueepochs.set('1000')
entreeepochs.pack(side = RIGHT, padx= 10)
zoneepochs.pack(fill="both", expand="yes", side = TOP)

#sous-cadre pour récupérer lr
zonelr = Frame(l2)
labellr = Label(zonelr, text="Pas d'apprentissage :")
labellr.pack(side= LEFT)
unit=Frame(zonelr)
valuelr = StringVar()
entreelr = Entry(unit, textvariable=valuelr, width=5)
valuelr.set('1')
entreelr.pack(side= LEFT, padx= 10)
labelunit = Label(unit, text="e-5")
labelunit.pack(side=RIGHT)
unit.pack(side=RIGHT)
zonelr.pack(fill="both", expand="yes", side = BOTTOM)


l2.pack(fill="both", expand="yes", side= TOP)

#sous-cadre pour récupérer Nb, N0, Nr
Ns= LabelFrame(cadredroit, text="paramètres de représentation", padx=20, pady=5)
topNs = Frame(Ns)
zoneNb = Frame(topNs)
labelNb = Label(zoneNb, text="Nombre de point aux limites :")
labelNb.pack(side= LEFT)
valueNb = StringVar()
entreeNb = Entry(zoneNb, textvariable=valueNb, width=5)
valueNb.set('500')
entreeNb.pack(side = RIGHT, padx= 10)
zoneNb.pack(fill="both", expand="yes", side = TOP)

zoneN0 = Frame(topNs)
labelN0 = Label(zoneN0, text="Points de conditions initiales :")
labelN0.pack(side= LEFT)
valueN0 = StringVar()
entreeN0 = Entry(zoneN0, textvariable=valueN0, width=5)
valueN0.set('500')
entreeN0.pack(side = RIGHT, padx= 10)
zoneN0.pack(fill="both", expand="yes", side = BOTTOM)

topNs.pack(side=TOP)

zoneNr = Frame(Ns)
labelNr = Label(zoneNr, text="Nombre de points résiduels :")
labelNr.pack(side= LEFT)
valueNr = StringVar()
entreeNr = Entry(zoneNr, textvariable=valueNr, width=5)
valueNr.set('500')
entreeNr.pack(side = RIGHT, padx= 10)
zoneNr.pack(fill="both", expand="yes", side = BOTTOM)

Ns.pack(side = BOTTOM)

cadredroit.pack(side=RIGHT)


top.pack(side=TOP)
bottom.pack(side= BOTTOM)

fenetre.mainloop()


"""def readequa(x, y=0, z=0):
    string = entreeequ.get()
    string = string.split(' ')
    r =[]
    for i in range(len(string)) :
        if string[i] == 'sin':
            r.append (np.sin)
        elif string[i] == 'cos':
            r.append(np.cos)
        elif 
        else :
            try : int(string[i])
            except : r.append
            else : """
        
    
    
    
    







