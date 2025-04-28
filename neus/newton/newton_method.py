import numpy as np

global_iteration_count = 0

def wolfe(f, g, xk, alpha, pk):
  c1 = 1e-4
  return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


def strong_wolfe(f, g, xk, alpha, pk, c2):
  # typically, c2 = 0.9 when using Newton or quasi-Newton's method.
  #            c2 = 0.1 when using non-linear conjugate gradient method.
  return wolfe(f, g, xk, alpha, pk) and abs(
      np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(np.dot(g(xk), pk))


def gold_stein(f, g, xk, alpha, pk, c):
  return (f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk)
          ) and (f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk))
def step_length(f, g, xk, alpha, pk, c2):
  return interpolation(f, g,
                       lambda alpha: f(xk + alpha * pk),
                       lambda alpha: np.dot(g(xk + alpha * pk), pk),
                       alpha, c2,
                       lambda f, g, alpha, c2: strong_wolfe(f, g, xk, alpha, pk, c2))


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha, iters=20):
  # referred implementation here:
  # https://github.com/tamland/non-linear-optimization
  l = 0.0
  h = 1.0
  for i in range(iters):
    if strong_wolfe_alpha(f, g, alpha, c2):
      return alpha

    half = (l + h) / 2
    alpha = - g_alpha(l) * (h**2) / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * h))
    if alpha < l or alpha > h:
      alpha = half
    if g_alpha(alpha) > 0:
      h = alpha
    elif g_alpha(alpha) <= 0:
      l = alpha
  return alpha


# optimization algorithms
def steepest_descent(f, grad, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in range(iterations):
    pk = -grad(x)
    alpha = step_length(f, grad, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 10 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
      print("  iter={}, x={}, f(x)={}".format(i, x, f(x)))

    if np.linalg.norm(x - x_old) < error:
      break
    x_old = x
  return x, i




def l_bfgs(f, g, x0, iterations, error, m=10):
  xk = x0
  c2 = 0.9
  I = np.identity(xk.size)
  Hk = I

  sks = []
  yks = []

  def Hp(H0, p):
    m_t = len(sks)
    q = g(xk)
    a = np.zeros(m_t)
    b = np.zeros(m_t)
    for i in reversed(range(m_t)):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      a[i] = rho_i * s.dot(q)
      q = q - a[i] * y

    r = H0.dot(q)

    for i in range(m_t):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      b[i] = rho_i * y.dot(r)
      r = r + s * (a[i] - b[i])

    return r

  for i in range(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hp(I, gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    sks.append(sk)
    yks.append(yk)
    if len(sks) > m:
      sks = sks[1:]
      yks = yks[1:]

    # compute H_{k+1} by BFGS update
    rho_k = float(1.0 / yk.dot(sk))

    if i % 10 == 0:
      print("  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, xk, f(xk)))
    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    xk = xk1

  return xk, i + 1
