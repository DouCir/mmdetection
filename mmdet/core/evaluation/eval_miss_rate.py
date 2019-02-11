import matlab.engine


def eval_miss_rate():
    eng = matlab.engine.start_matlab()
    eng.dbEval(nargout=0)
