import mitsuba as mi
import drjit as dr
from drjit.auto import Float, UInt, ArrayXf

# Indexing convention:
#
# order = 3
# for n in range(order+1):
#     for m in range(-n, n+1):
#         print(n, m)
#
# =====================
#    idx      n   m
# =====================
#     0:      0   0
#     1:      1   -1
#     2:      1   0
#     3:      1   1
#     4:      2   -2
#     5:      2   -1
#     6:      2   0
#     7:      2   1
#     8:      2   2
#     9:      3   -3
#     10:     3   -2
#     11:     3   -1
#     12:     3   0
#     13:     3   1
#     14:     3   2
#     15:     3   3

def get_sh_count(max_order: int) -> int:
    return (max_order + 1) ** 2

def get_sh_order_from_index(index: int) -> int:
    return int(dr.ceil(dr.sqrt(index + 1) - 1))

def eval_sh_at_position(shading_point_si, d, max_order: int = 3) -> mi.Color3f:
    sh_basis = dr.sh_eval(d, max_order)
    # NOTE: sh coefficients are themselves colors
    color = dr.zeros(mi.Color3f, dr.width(d))
    for sh_id, basis in enumerate(sh_basis):
        sh_coeff = shading_point_si.shape.eval_attribute_3(f"sh_coeffs_{sh_id}", shading_point_si)
        color += basis * sh_coeff
    return color

# ================================================
# Fitting
# ================================================

# def integrate_on_sphere(f, N: int = 256):
#     Ntheta = N
#     Nphi = Ntheta * 2
#     dtheta = dr.pi / Ntheta
#     dphi = 2.0 * dr.pi / Nphi
#     theta = dr.linspace(Float, 0.0, dr.pi, Ntheta, endpoint=False)
#     phi   = dr.linspace(Float, 0.0, 2.0 * dr.pi, Nphi, endpoint=False)
#     thetas, phis = dr.meshgrid(theta, phi)
#     d = mi.Vector3f(
#         dr.sin(thetas) * dr.cos(phis),
#         dr.sin(thetas) * dr.sin(phis),
#         dr.cos(thetas))
#     return dr.sum(f(d) * dr.sin(thetas) * dtheta * dphi, axis=None)

def spherical_integrate(f, N: int = 256) -> Float:
    '''
    Performs numerical integration of a scalar-valued function `f(\omega)` over the unit sphere.
    The integral is split into two 1D integrals over \theta (elevation) and \phi (azimuth):

    I = \int_{0}^{2\pi} \int_{0}^{\pi} f(\theta, \phi) * \sin{\theta} * d\theta * d\phi
     
    These integrals are remapped to the domain [-1,1] x [-1,1] so that standard quadrature rules 
    (compsoite Simpson, Gauss-Legendre, etc.) can be applied.
    '''
    # alternatives: quad.gauss_legendre, quad.gauss_lobatto
    nodes, weights = mi.quad.composite_simpson(N + 1)
    us, vs = dr.meshgrid(nodes, nodes)

    # W is the outer product of weights with itself, i.e. weights @ weights.T
    Nw = dr.width(weights)
    W = dr.gather(type(weights), weights, dr.tile(dr.arange(UInt, Nw), Nw))
    W *= dr.gather(type(W), weights, dr.repeat(dr.arange(UInt, Nw), Nw))

    thetas = 0.5 * dr.pi * (us + 1.0)
    phis = dr.pi * (vs + 1.0)
    d = mi.Vector3f(
        dr.sin(thetas) * dr.cos(phis),
        dr.sin(thetas) * dr.sin(phis),
        dr.cos(thetas))
    return dr.sum(W * f(d) * dr.sin(thetas), axis=None) * dr.square(dr.pi) * 0.5


def eval_basis(max_order: int, N: int = 256) -> tuple[mi.Vector3f, list[Float], Float]:
    '''
    Inputs:
        - max_order: int. The maximum SH degree to evaluate.
        - N: int. The number of directions to evaluate __per spherical angle__ (\theta, \phi). 
            A total of N*N directions will be queried to cover the full unit sphere.

    Outputs:
        - d: mi.Vector3f. Directions in which the SH basis functions are evaluated, size [3, N*N].
        - sh_basis: list[Float]. Evaluations of the SH basis functions in the directions `d`. The list
            contains `(max_order + 1) ** 2` entries -- one per basis function -- and each entry
            is a Float of size [N*N,].
        - W: Float. Quadrature weights matrix of size [N*N,].
    '''
    nodes_theta, weights_theta = mi.quad.composite_simpson(N // 2 + 1)
    nodes_phi, weights_phi = mi.quad.composite_simpson(N + 1)
    us, vs = dr.meshgrid(nodes_theta, nodes_phi)
    Nt = dr.width(weights_theta)
    Np = dr.width(weights_phi)
    W = dr.gather(type(weights_theta), weights_theta, dr.tile(dr.arange(UInt, Nt), Np))
    W *= dr.gather(type(W), weights_phi, dr.repeat(dr.arange(UInt, Np), Nt))
    thetas = 0.5 * dr.pi * (us + 1.0)
    phis = dr.pi * (vs + 1.0)
    st, ct = dr.sincos(thetas)
    sp, cp = dr.sincos(phis)
    d = mi.Vector3f(st * cp, st * sp, ct)

    sh_basis = dr.sh_eval(d, max_order)
    # absorb Jacobian term into quadrature weights "matrix"
    W *=  0.5 * dr.square(dr.pi) * dr.sin(thetas)
    return d, sh_basis, W

def eval_basis_on_hemisphere(max_order: int, N: int = 256) -> tuple[mi.Vector3f, list[Float], Float]:
    '''
    Variant of `eval_basis()` where we expect to evaluate a spherical function in the upper 
    hemisphere only. This is needed to fit *hemispheric* functions such as radiance caches
    or BRDFs; we essentially reflect the target function across the z-axis to yield a full 
    spherical function, and then compute its integral with the harmonic basis functions as 
    per usual.

    Inputs:
        - max_order: int. The maximum SH degree to evaluate.
        - N: int. The number of directions to evaluate __per spherical angle__ (\theta, \phi). 
            A total of N*N directions will be queried to cover the full unit sphere.

    Outputs:
        - d: mi.Vector3f. Directions in which the SH basis functions are evaluated, size [3, N*N].
        - sh_basis: list[Float]. Evaluations of the SH basis functions in the directions `d`. The list
            contains `(max_order + 1) ** 2` entries -- one per basis function -- and each entry
            is a Float of size [N*N,].
        - W: Float. Quadrature weights matrix of size [N*N,].
    '''
    nodes_theta, weights_theta = mi.quad.composite_simpson(N // 2 + 1)
    nodes_phi, weights_phi = mi.quad.composite_simpson(N + 1)
    us, vs = dr.meshgrid(nodes_theta, nodes_phi)
    Nt = dr.width(weights_theta)
    Np = dr.width(weights_phi)
    W = dr.gather(type(weights_theta), weights_theta, dr.tile(dr.arange(UInt, Nt), Np))
    W *= dr.gather(type(W), weights_phi, dr.repeat(dr.arange(UInt, Np), Nt))
    thetas = 0.5 * dr.pi * (us + 1.0)
    phis = dr.pi * (vs + 1.0)
    st, ct = dr.sincos(thetas)
    sp, cp = dr.sincos(phis)
    # Force the integrand-evaluated `z` directions into the upper hemisphere
    d = mi.Vector3f(st * cp, st * sp, dr.abs(ct))

    sh_basis = dr.sh_eval(d, max_order)
    # absorb Jacobian term into quadrature weights "matrix"
    W *=  0.5 * dr.square(dr.pi) * dr.sin(thetas)
    return d, sh_basis, W

def eval_basis_hemisphere_only(max_order: int, N: int = 256) -> tuple[mi.Vector3f, list[Float], Float]:
    '''
    Inputs:
        - max_order: int. The maximum SH degree to evaluate.
        - N: int. The number of directions to evaluate __per spherical angle__ (\theta, \phi). 
            A total of N*N directions will be queried to cover the full unit sphere.

    Outputs:
        - d: mi.Vector3f. Directions in which the SH basis functions are evaluated, size [3, N*N].
        - sh_basis: list[Float]. Evaluations of the SH basis functions in the directions `d`. The list
            contains `(max_order + 1) ** 2` entries -- one per basis function -- and each entry
            is a Float of size [N*N,].
        - W: Float. Quadrature weights matrix of size [N*N,].
    '''
    nodes_theta, weights_theta = mi.quad.composite_simpson(N // 4 + 1)
    nodes_phi, weights_phi = mi.quad.composite_simpson(N + 1)
    us, vs = dr.meshgrid(nodes_theta, nodes_phi)
    Nt = dr.width(weights_theta)
    Np = dr.width(weights_phi)
    W = dr.gather(type(weights_theta), weights_theta, dr.tile(dr.arange(UInt, Nt), Np))
    W *= dr.gather(type(W), weights_phi, dr.repeat(dr.arange(UInt, Np), Nt))
    thetas = 0.25 * dr.pi * (us + 1.0)
    phis = dr.pi * (vs + 1.0)
    st, ct = dr.sincos(thetas)
    sp, cp = dr.sincos(phis)
    d = mi.Vector3f(st * cp, st * sp, ct)

    sh_basis = dr.sh_eval(d, max_order)
    # absorb Jacobian term into quadrature weights "matrix"
    W *=  0.25 * dr.square(dr.pi) * dr.sin(thetas)
    return d, sh_basis, W

def fit_sh_coeffs_scalar(f_scalar, max_order: int, N: int = 64) -> ArrayXf:
    '''
    Compute the SH coefficients for a scalar-valued function `f` defined on the sphere.

    Inputs:
        - f_scalar: Function. The scalar-valued spherical function.
        - max_order: int. The maximum SH degree to evaluate.
        - N: int. The number of directions to evaluate __per spherical angle__ (\theta, \phi). 
            A total of N*N directions will be queried to cover the full unit sphere.

    Outputs:
        - sh_coeff: ArrayXf. The fitted SH coefficients, size [(max_order + 1) ** 2, 1].
    '''
    d, sh_basis, quad_W = eval_basis(max_order, N)
    integrand = quad_W * sh_basis * f_scalar(d)
    return dr.sum(integrand, axis=1)

def fit_sh_coeffs_color(f_color, max_order: int, N: int = 64):
    d, sh_basis, quad_W = eval_basis(max_order, N)
    f_evals = f_color(d)
    integrand_R = quad_W * sh_basis * f_evals.x
    integrand_G = quad_W * sh_basis * f_evals.y
    integrand_B = quad_W * sh_basis * f_evals.z
    I_R = dr.ravel(dr.sum(integrand_R, axis=1))
    I_G = dr.ravel(dr.sum(integrand_G, axis=1))
    I_B = dr.ravel(dr.sum(integrand_B, axis=1))
    return mi.Color3f(I_R, I_G, I_B)

def eval_sh_coeffs_color_on_sphere(sh_coeffs_: mi.Color3f, num_points: int, sh_order: int = None) -> mi.Color3f:
    sampler = mi.load_dict({'type': 'orthogonal'})
    sampler.seed(0, num_points)
    d = mi.warp.square_to_uniform_sphere(sampler.next_2d())
    if sh_order is None:
        sh_order = get_sh_order_from_index(dr.width(sh_coeffs_) - 1)
    
    if dr.width(sh_coeffs_) > get_sh_count(sh_order):
        sh_order = get_sh_order_from_index(dr.width(sh_coeffs_) - 1)

    sh_basis = dr.sh_eval(d, sh_order)      # shape: size-(sh_order+1)**2 list of Float[num_points, ]
    sh_coeffs = mi.TensorXf(dr.ravel(sh_coeffs_))
    color = dr.zeros(mi.Color3f, num_points)
    for index, sh in enumerate(sh_basis):
        color += sh * mi.Color3f(sh_coeffs[3*index:3*(index+1)])
    # color = dr.clip(spec, 0.0, 1.0)
    return color, d

def eval_sh_coeffs_color_for_direction(sh_coeffs_: mi.Color3f, d: mi.Vector3f):
    num_points = dr.width(d)
    sh_order = get_sh_order_from_index(dr.width(sh_coeffs_) - 1)

    sh_basis = dr.sh_eval(d, sh_order)      # shape: size-(sh_order+1)**2 list of Float[num_points, ]
    sh_coeffs = mi.TensorXf(dr.ravel(sh_coeffs_))
    color = dr.zeros(mi.Color3f, num_points)
    for index, sh in enumerate(sh_basis):
        color += sh * mi.Color3f(sh_coeffs[3*index:3*(index+1)])
    # color = dr.clip(spec, 0.0, 1.0)
    return color
