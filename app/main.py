from pathlib import Path
import argparse
from preprocessing import *

if __name__ == '__main__':
    path = Path.cwd()
    print(f"{str(path)}/files")
    print('PyCharm')
    example()
    example_2()
