import random as rng
import taichi as ti

# =========== USING TAICHI

ti.init(arch=ti.cpu) # when using only the cpu
# ti.init(arch=ti.cuda) #when using nvidia Gpu

N = 1000 # NxN grid
pixels = ti.field(dtype=ti.float32,shape=(N,N))
prev = ti.field(dtype=ti.float32,shape=(N,N))

for i in range(N):
    for j in range(N):
        prev[i,j]=rng.random()<.3

@ti.kernel
def paint():
    for x,y in ti.ndrange(N,N):
        alive_neighbours = 0
        for dx,dy in ti.ndrange(3,3):
            xx = x + dx -1
            yy = y + dy -1
            if 0 <= xx < N and 0 <= yy < N:
                alive_neighbours += prev[xx,yy]
        alive_neighbours -= prev[x,y]
        pixels[x,y] = 0.
        pixels[x,y] += prev[x,y] > .1 and 1.9 <= alive_neighbours <= 3.1
        pixels[x,y] += prev[x,y] < .1 and 2.9 < alive_neighbours < 3.1

gui=ti.GUI("Convays Game Of Life",res=(N,N))
while gui.running:
    paint()
    gui.set_image(pixels)
    gui.show()
    prev.copy_from(pixels)
