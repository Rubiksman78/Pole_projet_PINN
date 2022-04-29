import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

###Affichage du résultat dans une grille 2D ou 3D
def plot2d(lb,ub,N,tspace,model,fps):
    x1space = np.linspace(lb[1], ub[1], N + 1)
    x2space = np.linspace(lb[2], ub[2], N + 1)

    T,X1, X2 = np.meshgrid(tspace,x1space, x2space) #Découpage d'une grille de points
    Xgrid = tf.stack([T.flatten(),X1.flatten(),X2.flatten()],axis=-1)

    # Determine predictions of u(t, x)
    upred = model(tf.cast(Xgrid,DTYPE))

    U = upred.numpy().reshape(N+1,N+1,N+1)
    z_array = np.zeros((N+1,N+1,N+1))
    for i in range(N+1):
        z_array[:,:,i]= U[i]

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X1, X2,zarray[:,:,frame_number],cmap=cm.coolwarm,
                        linewidth=0)

    X1, X2 = np.meshgrid(x1space, x2space)
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(projection='3d')
    plot = [ax.plot_surface(X1, X2,z_array[:,:,0],cmap=cm.coolwarm,
                        linewidth=0, rstride=1, cstride=1)]
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    ax.set_zlabel('$u_\\theta(x1,x2)$')
    ax.set_zlim(0, 1.1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title('Solution of Wave equation')
    fig.colorbar(plot[0])

    plt.rcParams['animation.ffmpeg_path'] = "C:/SAMUEL/ffmpeg-5.0-full_build/ffmpeg-5.0-full_build/bin/ffmpeg.exe" #Faut installer un truc sombre pour faire l'animation vidéo, pas important
    anim = animation.FuncAnimation(fig,update_plot,N+1,fargs=(z_array,plot),interval=1000/fps)
    fn = 'plot_surface_animation_funcanimation'
    anim.save(fn+'.gif',writer='ffmpeg',fps=fps)
    plt.show()


def plot3d(lb,ub,N,tspace,model,fps):
    x1space = np.linspace(lb[1], ub[1], N + 1)
    x2space = np.linspace(lb[2], ub[2], N + 1)
    x3space = np.linspace(lb[3], ub[3], N + 1) #On échantillonne les composantes entre les bords avec N+1 points

    T,X1, X2,X3 = np.meshgrid(tspace,x1space, x2space,x3space) #Découpage d'une grille de points
    Xgrid = tf.stack([T.flatten(),X1.flatten(),X2.flatten(),X3.flatten()],axis=-1)

    # Determine predictions of u(t, x)
    upred = model(tf.cast(Xgrid,DTYPE))

    U = upred.numpy().reshape(N+1,N+1,N+1,N+1)
    z_array = np.zeros((N+1,N+1,N+1,N+1))
    for i in range(N+1):
        z_array[:,:,:,i]= U[i]

    def update_plot(frame_number, z_array, plot):
        plot[0].remove()
        plot[0] = ax.scatter(X1,X2,X3, c=z_array[:,:,:,frame_number],cmap='magma')

    X1, X2,X3 = np.meshgrid(x1space, x2space,x3space)
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111,projection='3d')
    plot = [ax.scatter(X1, X2,X3, c=z_array[:,:,:,0])]
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    ax.set_zlabel('$x3$')
    ax.set_title('Solution of Wave equation')

    plt.rcParams['animation.ffmpeg_path'] = "C:/SAMUEL/ffmpeg-5.0-full_build/ffmpeg-5.0-full_build/bin/ffmpeg.exe" #Faut installer un truc sombre pour faire l'animation vidéo, pas important
    anim = animation.FuncAnimation(fig,update_plot,N+1,fargs=(z_array,plot),interval=1000/fps)
    fn = 'plot_surface_animation_funcanimation'
    anim.save(fn+'.gif',writer='ffmpeg',fps=fps)
    plt.show()
    

def plot1d(lb,ub,N,tspace,model,fps):
    ###1D Wave
    x1space = np.linspace(lb[1], ub[1], N + 1)

    T,X1 = np.meshgrid(tspace,x1space)
    Xgrid = tf.stack([T.flatten(),X1.flatten()],axis=-1)

    upred = model(tf.cast(Xgrid,DTYPE))
    U = upred.numpy().reshape(N+1,N+1)
    z_array = np.zeros((N+1,N+1))
    for i in range(N):
        z_array[:,i]= U[i]

    plt.style.use('dark_background')

    fig = plt.figure()
    fig.set_dpi(100)
    ax1 = fig.add_subplot(1,1,1)

    def animate(k):
        x = z_array[:,k]
        k += 1
        ax1.clear()
        plt.plot(x1space,x,color='cyan')
        plt.ylim([-5,5])
        plt.xlim([0,1.0])
    plt.rcParams['animation.ffmpeg_path'] = "C:/SAMUEL/ffmpeg-5.0-full_build/ffmpeg-5.0-full_build/bin/ffmpeg.exe" #Faut installer un truc sombre pour faire l'animation vidéo, pas important
    anim = animation.FuncAnimation(fig,animate,frames=N,interval=20)
    fn = 'plot_1d_animation_funcanimation'
    anim.save(fn+'.gif',writer='ffmpeg',fps=fps)
    plt.show()