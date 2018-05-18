import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--x", "x", type=int, help="Step Size")
parser.add_argument("--y", "y", type=int, help="Step Size")
args = parser.parse_args()
k1 = args.x
k2 = args.y

print(" K1:", k1, " K2:  ", k2, " Sum ", k1 + k2) 
