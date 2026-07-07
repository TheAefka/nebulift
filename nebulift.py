import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)
os.chdir(SRC)

from main import main

if __name__ == "__main__":
    main()