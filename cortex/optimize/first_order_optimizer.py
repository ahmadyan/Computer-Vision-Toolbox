import numpy as np

def SGD(func, w_init, eta, steps):
    ws = [w_init]
    fs = [func(w_init, compute_grad=False)]
    
    for t in range(steps):
        w_cur = ws[-1]
        _, grad = func(w_cur, compute_grad=True)
        w_next = w_cur - eta*grad
        ws.append(w_next)
        fs.append(func(w_next, compute_grad=False))
    
    return np.concatenate(ws, axis=-1), np.array(fs).flatten()


def MomentumSGD(func, w_init, eta, steps, momentum=0.9, nesterov=False):
    ws = [w_init]
    fs = [func(w_init, compute_grad=False)]
    v = 0
    for t in range(steps):
        w_cur = ws[-1]
        _, grad = func(w_cur, compute_grad=True)
        v_prev = v
        v = momentum*v_prev - eta*grad
        if nesterov:
            w_next = w_cur - momentum*v_prev + (1+momentum)*v
        else:
            w_next = w_cur + v
        ws.append(w_next)
        fs.append(func(w_next, compute_grad=False))
    
    return np.concatenate(ws, axis=-1), np.array(fs).flatten()


def AdaGrad(func, w_init, eta, steps):
    ws = [w_init]
    fs = [func(w_init, compute_grad=False)]
    g = 0
    for t in range(steps):
        w_cur = ws[-1]
        _, grad = func(w_cur, compute_grad=True)
        g = g + grad**2
        w_next = w_cur - (eta/np.sqrt(g + 1e-8)) * grad
        ws.append(w_next)
        fs.append(func(w_next, compute_grad=False))
    
    return np.concatenate(ws, axis=-1), np.array(fs).flatten()


def RMSProp(func, w_init, eta, steps, gamma=0.9):
    ws = [w_init]
    fs = [func(w_init, compute_grad=False)]
    g = 0
    for t in range(steps):
        w_cur = ws[-1]
        _, grad = func(w_cur, compute_grad=True)
        g = gamma*g + (1-gamma)*grad**2
        w_next = w_cur - (eta/np.sqrt(g + 1e-8)) * grad
        ws.append(w_next)
        fs.append(func(w_next, compute_grad=False))
    
    return np.concatenate(ws, axis=-1), np.array(fs).flatten()


def Adam(func, w_init, eta, steps, beta1=0.9, beta2=0.999):
    ws = [w_init]
    fs = [func(w_init, compute_grad=False)]
    m = 0
    v = 0
    for t in range(1, steps+1):
        w_cur = ws[-1]
        _, grad = func(w_cur, compute_grad=True)
        m = beta1*m + (1-beta1)*grad
        mt = m/(1-beta1**t)
        v = beta2*v + (1-beta2)*grad**2
        vt = v/(1-beta2**t)
        w_next = w_cur - (eta/np.sqrt(v + 1e-8)) * m
        ws.append(w_next)
        fs.append(func(w_next, compute_grad=False))
    
    return np.concatenate(ws, axis=-1), np.array(fs).flatten()