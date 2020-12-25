import mxnet as mx
import numpy as np

def make_stdcdf(F, x, name="make_stdcdf"):
    return F.Custom(x, name=name, op_type="make_stdcdf")


class MakeDiagonalOp(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(MakeDiagonalOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n = a.shape[-1]
        if req[0] != 'null':
            if req[0] == 'write':
                b = out_data[0]
            else:
                b = mx.nd.zeros_like(out_data[0])
            index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int64)
            identity = mx.nd.one_hot(index, depth=n, dtype=a.dtype)
            dim_diff = len(b.shape) - len(identity.shape)
            if dim_diff > 0:
                res_shape = (1,)*dim_diff + identity.shape
                identity = mx.nd.reshape(identity, shape=res_shape)
            mx.nd.broadcast_to(identity, shape=out_data[0].shape, out=b)
            b *= mx.nd.expand_dims(a, axis=-1)
            if req[0] != 'write':
                self.assign(out_data[0], req[0], b)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        b_grad = out_grad[0]
        n = b_grad.shape[-1]
        # Extract diagonal of b_grad
        index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int64)
        identity = mx.nd.one_hot(index, depth=n, dtype=b_grad.dtype)
        dim_diff = len(b_grad.shape) - len(identity.shape)
        if dim_diff > 0:
            res_shape = (1,)*dim_diff + identity.shape
            identity = mx.nd.reshape(identity, shape=res_shape)
        bindex = mx.nd.broadcast_to(identity, shape=out_data[0].shape)
        a_grad = mx.nd.sum(b_grad*bindex, axis=-1)
        self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("make_diagonal")
class MakeDiagonalOpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MakeDiagonalOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['a']

    def list_outputs(self):
        return ['b']

    def infer_shape(self, in_shape):
        a_shape = in_shape[0]
        out_shape = a_shape[:-1]+[a_shape[-1], a_shape[-1]]
        return [a_shape], [out_shape], []

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return MakeDiagonalOp(**kwargs)


def make_diagonal(F, x, name="make_diagonal"):
    return F.Custom(x, name=name, op_type="make_diagonal")


class ExtractDiagonalOp(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(ExtractDiagonalOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n = a.shape[0]
        index = mx.nd.arange(start=0, stop=n * n, step=n + 1, dtype=np.int64)
        b = mx.nd.reshape(mx.nd.take(
            mx.nd.reshape(a, shape=(n * n,)), index), shape=(-1,))
        self.assign(out_data[0], req[0], b)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        b_grad = out_grad[0]
        n = b_grad.size
        if req[0] != 'null':
            index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int64)
            if req[0] == 'write':
                a_grad = in_grad[0]
            else:
                a_grad = mx.nd.zeros_like(in_grad[0])
            mx.nd.one_hot(index, depth=n, dtype=b_grad.dtype, out=a_grad)
            a_grad *= mx.nd.reshape(b_grad, shape=(-1, 1))
            if req[0] != 'write':
                self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("extract_diagonal")
class ExtractDiagonalOpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ExtractDiagonalOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['a']

    def list_outputs(self):
        return ['b']

    def infer_shape(self, in_shape):
        a_shape = in_shape[0]
        assert len(a_shape) == 2, 'a must have 2 dimensions'
        n = a_shape[0]
        assert a_shape[1] == n, 'a must be square matrix'

        return [a_shape], [(n,)], []

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return ExtractDiagonalOp(**kwargs)


def extract_diagonal(F, x, name="extract_diagonal"):
    return F.Custom(x, name=name, op_type="extract_diagonal")


class Identity_like_Op(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n = a.shape[0]
        if req[0] != 'null':
            index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int64)
            if req[0] == 'write':
                b = out_data[0]
            else:
                b = mx.nd.zeros_like(out_data[0])
            mx.nd.one_hot(index, depth=n, dtype=a.dtype, out=b)
            if req[0] != 'write':
                self.assign(out_data[0], req[0], b)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        a_grad = mx.nd.zeros_like(in_grad[0])
        self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("identity_like")
class Identity_like_OpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Identity_like_OpProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['a']

    def list_outputs(self):
        return ['b']

    def infer_shape(self, in_shape):
        a_shape = in_shape[0]
        assert len(a_shape) == 2, 'a must have 2 dimensions'
        n = a_shape[0]
        assert a_shape[1] == n, 'a must be square matrix'

        return [a_shape], [(n, n)], []

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return Identity_like_Op(**kwargs)


def identity_like(F, x, name="identity_like"):
    return F.Custom(x, name=name, op_type="identity_like")


def _positive_transform(F, x):
    return F.Activation(x, act_type='softrelu')


def _positive_transform_reverse(F, x):
    return F.log(F.expm1(x))


class LogSumExpOp(mx.operator.CustomOp):
    """Implementation of log sum exp for numerical stability
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        sum_x = mx.nd.sum(mx.nd.exp(x - max_x), axis=self.axis, keepdims=True)
        y = mx.nd.log(sum_x) + max_x
        y = y.reshape(out_data[0].shape)
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_grad[0]
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        y = y.reshape(max_x.shape)
        x = mx.nd.exp(x - max_x)
        prob = x / mx.nd.sum(x, axis=self.axis, keepdims=True)
        self.assign(in_grad[0], req[0], prob * y)


@mx.operator.register("log_sum_exp")
class LogSumExpProp(mx.operator.CustomOpProp):
    def __init__(self, axis, keepdims=False):
        super(LogSumExpProp, self).__init__(need_top_grad=True)
        self.axis = int(axis)
        self.keepdims = keepdims in ('True',)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        oshape = []
        for i, x in enumerate(data_shape):
            if i == self.axis:
                if self.keepdims:
                    oshape.append(1)
            else:
                oshape.append(x)
        return [data_shape], [tuple(oshape)], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogSumExpOp(self.axis)


def log_sum_exp(F, in_sym, axis, keepdims=False, name="log_sum_exp"):
    return F.Custom(in_sym, name=name, op_type="log_sum_exp",
                    axis=axis, keepdims=keepdims)


def backsub_both_sides(F, L, X, transpose=False):
    tmp = F.linalg.trsm(L, X, transpose=not transpose, rightside=True)
    return F.linalg.trsm(L, tmp, transpose=transpose)

        
def random_sample(bounds: np.ndarray, k: int) -> np.ndarray:
    """
    Generate a set of k n-dimensional points sampled uniformly at random
    :param bounds: n x 2 dimenional array containing upper/lower bounds for each dimension
    :param k: number of samples
    :return: k x n array containing the sampled points
    """
    
    # k: Number of points
    n = len(bounds)  # Dimensionality of each point
    X = np.zeros((k, n))
    for i in range(n):
        X[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], k)

    return X

def sample_Gaussian(F, mu, sigma):
    epsilon = F.random_normal(loc=0, scale=1, shape=mu.shape, dtype=mu.dtype)
    return mu + epsilon*sigma