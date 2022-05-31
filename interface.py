from calendar import c
from tkinter import *
import numpy as np
from pyparsing import col 
from pinn_code import *


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
    c,a,dimension,tmin,tmax,xmin,xmax,N_b,N_r,N_0,lr,epochs = int(entreec.get()), 0.5, int(entreedim.get()), 0., 1., 0., 1., 500, 500, 500, 1e-5, 10
    X_data,u_data,time_x,X_r = set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)
    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)
    plot1dgrid_real(lb,ub,200,lambda x: true_u(x,a,c),999999)
    opt = keras.optimizers.Adam(learning_rate=lr)
    hist = []
    pinn = PINN(dimension+1,1,dimension,ub,lb,c)
    pinn.compile(opt)
    train(epochs,pinn,X_r,X_data,u_data,true_u,N=100,dimension=dimension,batch_size=450)
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
#cadresortie = Frame(top, border = 2, width = 50)
#cadresortie.pack(side=RIGHT)

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
        
    
    
    
    







