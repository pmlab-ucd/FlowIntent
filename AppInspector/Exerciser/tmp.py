from nodebox.graphics import *
from nodebox.graphics.physics import Flock

graph =  ximport("graph")
g = graph.create(iterations=500, distance=0.8)
g.add_node("NodeBox")
g.add_node("Core Image", category="library")
g.add_edge("Core Image", "NodeBox")
g.solve()
g.draw()