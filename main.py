import numpy as np
import matplotlib.pyplot as plt

### Prob definition
# Vertical pipe
# x is vertical direction from 0 to length
# y is horizontal from -1/2 diameter to 1/2 diameter
# Since its radially heated a 2D slice can represent everything

### Assumptions
# Steady state (d/dt = 0)
# Incompressible flow (dv = 0)
# No-slip condition (v=0 at walls)
# Flow enters at T=Tmelt
# Walls remain 565 degrees
# gravity is earths
# The fluid in the pipe starts at 285
# Numan boundary conditions at the outlet

### Equasions
# vx (dvx/dx) + vy (dvx/dy) = v (d^2vx/dy^2) + gB(T-Tm)
# x-momentum equation, gravity acts along x, Tmelt is the inlet temperature

# vx (dT/dx) + vy (dT/dy) = a (d^2T/dy^2)
# energy equation, convects upwards

# dvx/dx + dvy/dy = 0
# Continuity equation, from the incompressible flow assumption

### Boundry conditions
# Twall = 565
# vx=vy = 0 at wall
# Tmelt = 285
# dT/dx=dvx/dx=dvy/dx=0 at outlet





# Conditions
l = 5.1                 # pipe length [m]
d = 0.0206              # pipe diameter [m]
Tw = 565.0              # wall temperature [C]
Tm = 285.0              # inlet temperature [C]
alpha = 1.90*(10**-7)   # thermal diffusivity [m^2/s]
g = 9.81                # gravity [m/s^2]
B = 3.6 * (10**-4)      # thermal expansion [1/K]
p = 1822.88             # density [kg/m^3]
n = 0.00263             # viscosity [kg/ms]


# Resolution
dx = 0.01       # how fine x resolution is [m]
dy = 0.001       # how fine y resolution is [m]

Dx = int(l/dx) + 2 # size + 2 walls
Dy = int(d/dy) + 2 # size + 2 walls

# Makes the grids
T = np.ones((Dx, Dy)) * Tm        # temperature
vx = np.zeros((Dx, Dy))           # no vertical flow [m/s]
vy = np.zeros((Dx, Dy))           # no horizontal flow [m/s]

# Define the wall temp
T[:, 0] = Tw       # left wall
T[:, -1] = Tw      # right wall

# Solve
maxIteration = 1000
relaxationFactor = 0.01

for iteration in range(maxIteration):
    print(maxIteration - iteration)
    TPrevious = T.copy()
    vxPrevious = vx.copy()
    vyPrevious = vy.copy()
    for i in range(1, Dx-1):
        for j in range(1, Dy-1):
            vxTerm = vxPrevious[i, j] * (TPrevious[i+1, j] - TPrevious[i-1, j]) / (2*dx)
            vyTerm = vyPrevious[i, j] * (TPrevious[i, j+1] - TPrevious[i, j-1]) / (2*dy)
            thermalDiffusion = alpha * ((TPrevious[i, j+1] - 2*TPrevious[i, j] + TPrevious[i, j-1]) / (dy) / (dy))

            T[i, j] = TPrevious[i, j] + relaxationFactor * (thermalDiffusion - vxTerm - vyTerm)

            vxConv = vxPrevious[i, j] * (vxPrevious[i+1, j]-vxPrevious[i-1, j]) / (2*dx)
            vyConv = vyPrevious[i, j] * (vxPrevious[i, j+1]-vxPrevious[i, j-1]) / (2*dy)
            vxDiffusion = (n/p) * ((vxPrevious[i, j+1] - 2*vxPrevious[i,j] + vxPrevious[i, j-1]) / (dy) / (dy))
            buoyancy = g * B * (TPrevious[i,j]-Tm)

            vx[i, j] = vxPrevious[i, j] + relaxationFactor * (buoyancy + vxDiffusion - vyConv - vxConv)


    vx[-1, :] = vx[-2, :]
    vx = np.clip(vx, -0.13, 0.13)

# with open('temp.txt', 'w') as file:
#     for i in T:
#         file.write(str(i) + '\n')

plt.figure(figsize=(8, 4))
plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx), T, cmap='hot', vmin=Tm, vmax=Tw, levels=1000)
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Temperature Field in Pipe, Resolution: ' + str(dy*100) + ' cm')


plt.figure(figsize=(8, 4))
plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx), vx, cmap='bone', levels=np.linspace(0, 0.06, 1000), vmin=0, vmax=0.06)
plt.colorbar(label='Speed (m/s)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Upwards Velocity Field in Pipe, Resolution: ' + str(dy*100) + ' cm')


size = 5
plt.figure(figsize=(8, 4))
plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:], cmap='hot', vmin=Tm, vmax=Tw, levels=900)
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Temperature Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c1 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:], cmap='hot', vmin=Tm, vmax=Tw, levels=900)
plt.contour(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:],levels=[1.005*Tm], colors='green', linestyles='dashed', linewidths=2)
plt.colorbar(c1, label='Temperature (°C)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Temperature Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c2 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:], cmap='hot', vmin=Tm, vmax=Tw, levels=900)
plt.contour(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:],levels=[1.005*Tm], colors='green', linestyles='dashed', linewidths=2)
plt.axhline(y=0.023, color='blue', linestyle='dashed', label="2.30 cm")
plt.axvline(x=-0.767, color='blue', linestyle='dashed', label="0.767 cm")
plt.axvline(x=0.767, color='blue', linestyle='dashed', label="-0.767 cm")
plt.colorbar(c2, label='Temperature (°C)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Temperature Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c3 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:], cmap='flag', vmin=Tm, vmax=Tw, levels=900)
plt.contour(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], T[:size,:],levels=[1.005*Tm], colors='green', linestyles='dashed', linewidths=2)
plt.axhline(y=0.023, color='blue', linestyle='dashed', label="2.30 cm")
plt.axvline(x=-0.767, color='blue', linestyle='dashed', label="0.767 cm")
plt.axvline(x=0.767, color='blue', linestyle='dashed', label="-0.767 cm")
plt.colorbar(c3, label='Temperature (°C)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Temperature Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c4 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], vx[:size,:], cmap='bone', levels=np.linspace(0, 0.06, 1000), vmin=0, vmax=0.06)
plt.colorbar(c4, label='Speed (m/s)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Upwards Velocity Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c5 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], vx[:size,:], cmap='bone', levels=np.linspace(0, 0.06, 1000), vmin=0, vmax=0.06)
plt.contour(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], vx[:size,:],levels=[0.005*0.06], colors='green', linestyles='dashed', linewidths=2)
plt.colorbar(c5, label='Speed (m/s)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Upwards Velocity Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')

plt.figure(figsize=(8, 4))
c5 = plt.contourf(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], vx[:size,:], cmap='bone', levels=np.linspace(0, 0.06, 1000), vmin=0, vmax=0.06)
plt.contour(np.linspace(-d/2,d/2,Dy)*100, np.linspace(0,l,Dx)[0:size], vx[:size,:],levels=[0.005*0.06], colors='green', linestyles='dashed', linewidths=2)
plt.axhline(y=0.0105, color='blue', linestyle='dashed', label="1.05 cm")
plt.axvline(x=-0.552, color='blue', linestyle='dashed', label="0.552 cm")
plt.axvline(x=0.552, color='blue', linestyle='dashed', label="-0.552 cm")
plt.colorbar(c5, label='Speed (m/s)')
plt.xlabel('Radius [cm]')
plt.ylabel('Length [m]')
plt.title('Upwards Velocity Boundary Layer Formation in Pipe, Resolution: ' + str(dy*100) + ' cm')
plt.show()