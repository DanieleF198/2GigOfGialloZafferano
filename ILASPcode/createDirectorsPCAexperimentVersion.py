import os
import shutil
from datetime import datetime

PCAindexes = [5, 10, 15, 20]
COUPLES = [45, 105, 210]
scopes = ["", "_original"]
for scope in scopes:
    for PCAindex in PCAindexes:
        for COUPLE in COUPLES:
            pathFinalDirZeroTrain = './PCAexperiment/final' + scope + str(PCAindex) + '/users/zero/train/'
            pathFinalDirZeroTest = './PCAexperiment/final' + scope + str(PCAindex) + '/users/zero/test/'
            pathFinalDirNoZeroTrain = './PCAexperiment/final' + scope + str(PCAindex) + '/users/no_zero/train/'
            pathFinalDirNoZeroTest = './PCAexperiment/final' + scope + str(PCAindex) + '/users/no_zero/test/'
            
            if scope == "_original":
                pathUsersZeroCoupleTrainString = pathFinalDirZeroTrain + '150Couples'
                pathUsersZeroCoupleTestString = pathFinalDirZeroTest + '50Couples'
                pathUsersNoZeroCoupleTrainString = pathFinalDirNoZeroTrain + '150Couples'
                pathUsersNoZeroCoupleTestString = pathFinalDirNoZeroTest + '50Couples'
            else:
                pathUsersZeroCoupleTrainString = pathFinalDirZeroTrain + str(COUPLE) + 'Couples'
                pathUsersZeroCoupleTestString = pathFinalDirZeroTest + '105Couples'
                pathUsersNoZeroCoupleTrainString = pathFinalDirNoZeroTrain + str(COUPLE) + 'Couples'
                pathUsersNoZeroCoupleTestString = pathFinalDirNoZeroTest + '105Couples'
            
            effectivePathUsersZeroCoupleTrainString = os.path.join(pathUsersZeroCoupleTrainString)
            effectivePathUsersZeroCoupleTestString = os.path.join(pathUsersZeroCoupleTestString)
            effectivePathUsersNoZeroCoupleTrainString = os.path.join(pathUsersNoZeroCoupleTrainString)
            effectivePathUsersNoZeroCoupleTestString = os.path.join(pathUsersNoZeroCoupleTestString)

            # os.makedirs(effectivePathUsersZeroCoupleTrainString, exist_ok=True)
            # os.makedirs(effectivePathUsersZeroCoupleTestString, exist_ok=True)
            # os.makedirs(effectivePathUsersNoZeroCoupleTrainString, exist_ok=True)
            # os.makedirs(effectivePathUsersNoZeroCoupleTestString, exist_ok=True)
            
            for i in range(0, 54):
                pathUsersZeroFinalTrainString = pathUsersZeroCoupleTrainString + '/User' + str(i)
                pathUsersZeroFinalTestString = pathUsersZeroCoupleTestString + '/User' + str(i)
                pathUsersNoZeroFinalTrainString = pathUsersNoZeroCoupleTrainString + '/User' + str(i)
                pathUsersNoZeroFinalTestString = pathUsersNoZeroCoupleTestString + '/User' + str(i)
            
                effectivePathUsersZeroFinalTrainString = os.path.join(pathUsersZeroFinalTrainString)
                effectivePathUsersZeroFinalTestString = os.path.join(pathUsersZeroFinalTestString)
                effectivePathUsersNoZeroFinalTrainString = os.path.join(pathUsersNoZeroFinalTrainString)
                effectivePathUsersNoZeroFinalTestString = os.path.join(pathUsersNoZeroFinalTestString)
            
                os.makedirs(effectivePathUsersZeroFinalTrainString, exist_ok=True)
                os.makedirs(effectivePathUsersZeroFinalTestString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroFinalTrainString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroFinalTestString, exist_ok=True)
            
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
            
                os.makedirs(effectivePathUsersZeroFilesTrainFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersZeroFilesTestFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroFilesTrainFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroFilesTestFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersZeroOutputTrainFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersZeroOutputTestFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroOutputTrainFinalString, exist_ok=True)
                os.makedirs(effectivePathUsersNoZeroOutputTestFinalString, exist_ok=True)
