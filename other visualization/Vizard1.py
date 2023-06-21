import viz
import vizact

viz.setMultiSample(4)
viz.go()

viz.MainWindow.fov(60)

piazza = viz.addChild('piazza.osgb')
# viz.MainView.collision(viz.ON)
viz.MainView.move([3,0,-7])

plants = []
for x in [-3, -1, 1, 3]:
    for z in [4, 2, 0, -2, -4]:
        plant = viz.addChild('plant.osgb',cache=viz.CACHE_CLONE)
        plant.setPosition([x,0,z])
        plants.append(plant)