import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", help="choose the velocity of the wave", type = float, default=2)
parser.add_argument("-dim", help= "dimension of the modelisation", choices=[1, 2, 3], type=int, default=1)
parser.add_argument("-Nb", help = "number of points at the edge", type = int, default=500)
parser.add_argument("-N0", help = "number of initial points", type = int, default=500)
parser.add_argument("-Nr", help = "number of residual points", type = int, default=500)
parser.add_argument("-tmin", help = "lower bound of time simulation",type = float,default=0.)
parser.add_argument("-tmax", help = "upper bound of time simulation",type = float,default=1.)
parser.add_argument("-xmin", help = "lower bound of space simulation",type = float,default=0.)
parser.add_argument("-xmax", help = "upper bound of space simulation",type = float,default=1.)
parser.add_argument("-epochs", help = "number of epochs for simulation",type = int,default=100000)
args = parser.parse_args()

def define_config():
    config = {
    'c' : args.c,
    'a':0.5,
    'dimension' : args.dim,
    'tmin' : args.tmin,
    'tmax' : args.tmax,
    'xmin': args.xmin,
    'xmax': args.xmax,
    'N_b' : args.Nb,
    'N_0' : args.N0,
    'N_r' : args.Nr,
    'learning_rate' : 1e-5,
    'epochs': args.epochs
    }
    return config