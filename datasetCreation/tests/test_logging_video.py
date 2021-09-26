import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import cv2
import pybullet_robots.panda.panda_sim_grasp as panda_sim

# video requires ffmpeg available in path
createVideo = True
fps = 240.
timeStep = 1. / fps

if createVideo:
    p.connect(p.GUI)#, options="--minGraphicsUpdateTimeMs=0 --mp4=\"video.mp4\" --mp4fps=" + str(fps))
else:
    p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, -0.13, 0])
p.setAdditionalSearchPath(pd.getDataPath())

p.setTimeStep(timeStep)
p.setGravity(0, -9.8, 0)

panda = panda_sim.PandaSimAuto(p, [0, 0, 0])
panda.control_dt = timeStep

# logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
# logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_VIDEO_MP4, "video.mp4")
panda.bullet_client.submitProfileTiming("start")
for i in range(1):
    panda.bullet_client.submitProfileTiming("full_step")
    panda.step()
    p.stepSimulation()
    output = p.getCameraImage(1000, 1000)
    rgb = np.array(output[2]).reshape((1000, 1000, 4))[:, :, :3]
    img = rgb.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("rgb_image.png", img)
    if createVideo:
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    if not createVideo:
        time.sleep(timeStep)
    panda.bullet_client.submitProfileTiming()
panda.bullet_client.submitProfileTiming()
panda.bullet_client.stopStateLogging(logId)
