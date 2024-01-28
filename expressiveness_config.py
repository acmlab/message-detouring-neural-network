import sys
DEN_K = int(sys.argv[1])
method, K = 'WL', 0
MAX_ITER = 10
USE_DEN = sys.argv[2]=="1"
PARALLEL_DEN = False
kwl_jobn = 80
DEN_HASH_LEVEL = 2 # 0, 1, 2
ONLY_INIT = False
mtag = '' if not USE_DEN else f'-DeN{DEN_K}'
mtag = mtag.replace('DeN', f'H{DEN_HASH_LEVEL}DeN') if DEN_HASH_LEVEL > 0 else mtag