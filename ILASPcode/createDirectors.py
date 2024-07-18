import os
from datetime import datetime


pathFinalDirZeroTrain = './Data17Component2Std/final_original/users/zero/train/'
pathFinalDirZeroTest = './Data17Component2Std/final_original/users/zero/test/'
pathFinalDirNoZeroTrain = './Data17Component2Std/final_original/users/no_zero/train/'
pathFinalDirNoZeroTest = './Data17Component2Std/final_original/users/no_zero/test/'

pathUsersZeroCoupleTrainString = pathFinalDirZeroTrain + '150Couples'
pathUsersZeroCoupleTestString = pathFinalDirZeroTest + '50Couples'
pathUsersNoZeroCoupleTrainString = pathFinalDirNoZeroTrain + '150Couples'
pathUsersNoZeroCoupleTestString = pathFinalDirNoZeroTest + '50Couples'

effectivePathUsersZeroCoupleTrainString = os.path.join(pathUsersZeroCoupleTrainString)
effectivePathUsersZeroCoupleTestString = os.path.join(pathUsersZeroCoupleTestString)
effectivePathUsersNoZeroCoupleTrainString = os.path.join(pathUsersNoZeroCoupleTrainString)
effectivePathUsersNoZeroCoupleTestString = os.path.join(pathUsersNoZeroCoupleTestString)
#
os.makedirs(effectivePathUsersZeroCoupleTrainString)
os.makedirs(effectivePathUsersZeroCoupleTestString)
os.makedirs(effectivePathUsersNoZeroCoupleTrainString)
os.makedirs(effectivePathUsersNoZeroCoupleTestString)

for i in range(0, 54):
    pathUsersZeroFinalTrainString = pathUsersZeroCoupleTrainString + '/User' + str(i)
    pathUsersZeroFinalTestString = pathUsersZeroCoupleTestString + '/User' + str(i)
    pathUsersNoZeroFinalTrainString = pathUsersNoZeroCoupleTrainString + '/User' + str(i)
    pathUsersNoZeroFinalTestString = pathUsersNoZeroCoupleTestString + '/User' + str(i)

    effectivePathUsersZeroFinalTrainString = os.path.join(pathUsersZeroFinalTrainString)
    effectivePathUsersZeroFinalTestString = os.path.join(pathUsersZeroFinalTestString)
    effectivePathUsersNoZeroFinalTrainString = os.path.join(pathUsersNoZeroFinalTrainString)
    effectivePathUsersNoZeroFinalTestString = os.path.join(pathUsersNoZeroFinalTestString)

    os.makedirs(effectivePathUsersZeroFinalTrainString)
    os.makedirs(effectivePathUsersZeroFinalTestString)
    os.makedirs(effectivePathUsersNoZeroFinalTrainString)
    os.makedirs(effectivePathUsersNoZeroFinalTestString)

    pathUsersZeroFilesTrainFinalString = effectivePathUsersZeroFinalTrainString + '/trainFiles'
    pathUsersZeroFilesTestFinalString = effectivePathUsersZeroFinalTestString + '/testFiles'
    pathUsersNoZeroFilesTrainFinalString = effectivePathUsersNoZeroFinalTrainString + '/trainFiles'
    pathUsersNoZeroFilesTestFinalString = effectivePathUsersNoZeroFinalTestString + '/testFiles'
    pathUsersZeroOutputTrainFinalString = effectivePathUsersZeroFinalTrainString + '/outputTrain'
    pathUsersZeroOutputTestFinalString = effectivePathUsersZeroFinalTestString + '/outputTest'
    pathUsersNoZeroOutputTrainFinalString = effectivePathUsersNoZeroFinalTrainString + '/outputTrain'
    pathUsersNoZeroOutputTestFinalString = effectivePathUsersNoZeroFinalTestString + '/outputTest'

    effectivePathUsersZeroFilesTrainFinalString = os.path.join(pathUsersZeroFilesTrainFinalString)
    effectivePathUsersZeroFilesTestFinalString = os.path.join(pathUsersZeroFilesTestFinalString)
    effectivePathUsersNoZeroFilesTrainFinalString = os.path.join(pathUsersNoZeroFilesTrainFinalString)
    effectivePathUsersNoZeroFilesTestFinalString = os.path.join(pathUsersNoZeroFilesTestFinalString)
    effectivePathUsersZeroOutputTrainFinalString = os.path.join(pathUsersZeroOutputTrainFinalString)
    effectivePathUsersZeroOutputTestFinalString = os.path.join(pathUsersZeroOutputTestFinalString)
    effectivePathUsersNoZeroOutputTrainFinalString = os.path.join(pathUsersNoZeroOutputTrainFinalString)
    effectivePathUsersNoZeroOutputTestFinalString = os.path.join(pathUsersNoZeroOutputTestFinalString)

    os.makedirs(effectivePathUsersZeroFilesTrainFinalString)
    os.makedirs(effectivePathUsersZeroFilesTestFinalString)
    os.makedirs(effectivePathUsersNoZeroFilesTrainFinalString)
    os.makedirs(effectivePathUsersNoZeroFilesTestFinalString)
    os.makedirs(effectivePathUsersZeroOutputTrainFinalString)
    os.makedirs(effectivePathUsersZeroOutputTestFinalString)
    os.makedirs(effectivePathUsersNoZeroOutputTrainFinalString)
    os.makedirs(effectivePathUsersNoZeroOutputTestFinalString)
