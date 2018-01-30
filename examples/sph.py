import points.points
import numpy as np

# Example (parameters and setup) based on Saeed Mahani's 594G final project
# github link: https://github.com/saeedmahani/SPH-Fluid-Simulation
N = 3000				# number of particles
h = 0.0457			# support radius (m)
m = 0.02				# particle mass (kg)
g = 9.80665			# gravity acceleration (m/s^2)
K = 3.0				# gas stiffness of water vapor (Nm/kg)
mu = 3.5				# viscosity of water (Ns/m^2)
sigma = 0.0728		# surface tension coeffcient (N/m)
L = 7.065			# threshold on magnitude of color field normal
rho_o = 998.29		# rest density (kg/m^3)
k_wall = 1e5		# spring constant of wall (N/m)
c_wall = -0.9		# damping constant of wall

# particles confined to "infinite square well:
# [-0.2,+0.2]x[-0.2,inf]x[-0.2,+0.2], where +y is up
n_wall = np.matrix('\
	+1,0,0;\
	-1,0,0;\
	0,+1,0;\
	0,0,+1;\
	0,0,-1')
pos_wall = np.matrix('-1;+1;-1;-1;+1')

# smoothing kernel functions and their derivatives
def WspikyGradient(x):
	r = np.sqrt(veclen2(x))
	return -45/(np.pi*np.power(h,6))*\
		np.multiply(np.square(h-r),np.divide(x,r))
def Wpoly6(x):
	return 315/(64*np.pi*np.power(h,9))*\
		np.power(np.square(h)-veclen2(x),3)
def Wpoly6Gradient(x):
	return -945/(32*np.pi*np.power(h,9))*\
		np.multiply(np.square(h)-veclen2(x),x)
def Wpoly6Laplacian(x):
	return -945/(32*np.pi*np.power(h,9))*\
		np.multiply(np.square(h)-veclen2(x),3*np.square(h)-7*veclen2(x))
def WviscocityLaplacian(x):
	return 45/(np*pi*np.power(h,6))*(h-np.sqrt(veclen2(x)))

# convenience function for computing squared length
def veclen2(x):
	return np.sum(np.square(x),axis=1)

# initialize particles and allocate scratch space
P = points.points(numpoints=N)	# point positions
# todo: initialize particle positions

# allocate scratch space
Ps = []
V = np.matrix(N,3)		# point velocities
A = np.matrix(N,3)		# point accelerations
rho = np.matrix(N,1)
p = np.matrix(N,1)

# begin simulation loop
dt = 0.01
for t in np.linsapce(0,1,dt):
	Ps.append(P)	# need to make sure this appends a copy of P and not a shared reference

	# compute neighborhoods
	nhbrhoods = P.nhbrhoods(r=h)

	# compute density and pressure at each particle
	for i,indices in enumerate(nhbrhoods):
		rho[i] = m*np.sum(Wpoly6(P[indices]-P[i]))
		p[i] = K*(rho[i]-rho_o)

	# compute forces at each particle
	for i,indices in enumerate(nhbrhoods):
		dx = P[indices]-P[i]

		# pressure force
		A[i] = -m*np.sum(np.multiply(\
			np.divide(p[i]+p[indices],2*rho[indices]),\
			WspikyGradient(dx)),axis=0)

		# viscosity force
		A[i] += mu*m*np.sum(np.multiply(\
			np.divide(V[indices]-V[i],rho[indices]),
			WviscocityLaplacian(dx)),axis=0)

		# tension force
		n = m*np.sum(np.multiply(\
			np.divide(1,rho[indices]),
			Wpoly6Gradient(dx)),axis=0)
		if veclen2(n) > L:
			A[i] += -sigma*np.divide(n,np.sqrt(veclen2(n)))
				m*np.sum(np.multiply(\
				1/rho[indices],
				Wpoly6Laplacian(dx)))

		A[i] /= rho[i]

		# wall penetration depth
		d = np.minimum(0,-n_wall*P[i].T+pos_wall+0.01)

		# wall force
		A[i] += k_wall*np.sum(np.multiply(d,n_wall),axis=0)
		A[i] += c_wall*np.sum(np.multiply(n_wall*V[i].T,n_wall),axis=0)

	# advance particles
	P += V*dt+A*dt*dt
	V = (P-P[-1])/dt	

# append results of last time step
Ps.append(P)
