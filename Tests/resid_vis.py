import numpy as np
import matplotlib.pyplot as plt

# Symmetric positive definite matrix defining the quadratic form
A = np.array([[5.0, 2.0],
              [2.0, 1.0]])

b = np.array([1.0, 0.0])

# initial guess
x = np.array([-4.0, 4.0])

# lists to store the iterates and residuals
xs = [x.copy()]
rs = []

# perform a few iterations of (un-preconditioned) Steepest Descent
max_iter = 5
for k in range(max_iter):
    r = b - A @ x                        # residual (negative gradient)
    rs.append(r.copy())
    alpha = (r @ r) / (r @ A @ r)        # optimal step length
    x = x + alpha * r                    # update iterate
    xs.append(x.copy())

xs = np.array(xs)
rs = np.array(rs)

# create a contour plot of the quadratic ϕ(x)=½ xᵀAx - bᵀx
xx, yy = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-2, 6, 400))
Z = 0.5 * (A[0,0]*xx**2 + 2*A[0,1]*xx*yy + A[1,1]*yy**2) - (b[0]*xx + b[1]*yy)

plt.figure(figsize=(6, 5))
# contour lines
plt.contour(xx, yy, Z, levels=30, linewidths=0.6)

# path of the iterates
plt.plot(xs[:,0], xs[:,1], marker='o')

# draw residual (search) directions at each step for intuition
for i in range(max_iter):
    p = rs[i]                            # search direction (same as residual)
    origin = xs[i]
    plt.arrow(origin[0], origin[1], 0.3*p[0], 0.3*p[1],
              head_width=0.15, length_includes_head=True)

plt.title("Steepest Descent: consecutive orthogonal residuals")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True)
plt.axis('equal')
plt.show()
