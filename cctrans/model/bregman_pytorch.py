import torch

M_EPS = 1e-16


def sinkhorn_knopp(
    a,
    b,
    C,
    reg=1e-1,
    maxIter=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warm_start=None,
    eval_freq=10,
    print_freq=200,
    **kwargs
):
    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, " C needs to be 2d"
    assert (
        na == a.shape[0] and nb == b.shape[0]
    ), "Shape of a or b does`t match that of C"
    assert reg > 0, "reg should be greater than 0"
    assert a.min() >= 0.0 and b.min() >= 0.0, "Elements in a or b less than 0"

    if log:
        log = {"err": []}

    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    while err > stopThr and it <= maxIter:
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        if (
            torch.any(torch.isnan(u))
            or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u))
            or torch.any(torch.isinf(v))
        ):
            print("Warning: numerical errors at iteration", it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log["err"].append(err)

        if verbose and it % print_freq == 0:
            print("iteration {:5d}, constraint error {:5e}".format(it, err))

        it += 1
    if log:
        log["u"] = u
        log["v"] = v
        log["alpha"] = reg * torch.log(u + M_EPS)
        log["beta"] = reg * torch.log(v + M_EPS)

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)

    if log:
        return P, log
    else:
        return P


if __name__ == "__main__":
    import numpy as np

    a = torch.from_numpy(np.array([3, 3, 3, 4, 2, 2, 2, 1]).astype('float32'))
    b = torch.from_numpy(np.array([4, 2, 6, 4, 4]).astype('float32'))
    C = torch.from_numpy(
        np.array(
            [
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, -2.0, -2.0, -2.0, 2.0],
                [1.0, 2.0, 2.0, 2.0, -1.0],
                [2.0, 1.0, 0.0, 1.0, -1.0],
                [0.5, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, -1.0],
                [-2.0, 2.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 2.0, 1.0, -1.0],
            ]
        ).astype('float32')
    )
    P, log = sinkhorn_knopp(a,b,C, log= True, reg= 10, maxIter= 100)
    print(np.round(P.numpy()))
    print(log['beta'])