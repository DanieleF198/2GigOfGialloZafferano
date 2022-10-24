import os
from datetime import datetime


pathPlotsBaseString = './NN_data/plots/'
pathModelBaseString = './NN_data/models/'
now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
# year = "2022"
# month = "01"
# day = "23"
data_dir = "./dataset_100/separated_text_data/"


for i in range(0, 48):
    pathPlotsFinalString = pathPlotsBaseString + 'User' + str(i)
    pathModelBaseFinalString = pathModelBaseString + 'User' + str(i)

    effectivePathPlots = os.path.join(pathPlotsFinalString)
    effectivePathModel = os.path.join(pathModelBaseFinalString)
    effectivePathModelNormal = os.path.join(pathModelBaseFinalString + '/folder_version')
    # effectivePathModelGauss = os.path.join(pathModelBaseFinalString + '/resultGauss')

    if not os.path.exists(effectivePathPlots):
        os.makedirs(effectivePathPlots)
    if not os.path.exists(effectivePathModel):
        os.makedirs(effectivePathModel)
    if not os.path.exists(effectivePathModelNormal):
        os.makedirs(effectivePathModelNormal)
    # if not os.path.exists(effectivePathModelGauss):
    #     os.makedirs(effectivePathModelGauss)

    pathPlotsAccuracyFinalString = pathPlotsFinalString + '/accuracy'
    # pathPlotsAccuracyGaussFinalString = pathPlotsFinalString + '/accuracy/resultGauss'
    pathPlotsLossFinalString = pathPlotsFinalString + '/loss'
    # pathPlotsLossGaussFinalString = pathPlotsFinalString + '/loss/resultGauss'
    pathPlotsPrecisionFinalString = pathPlotsFinalString + '/precision'
    # pathPlotsPrecisionGaussFinalString = pathPlotsFinalString + '/precision/resultGauss'
    pathPlotsRecallFinalString = pathPlotsFinalString + '/recall'
    # pathPlotsRecallGaussFinalString = pathPlotsFinalString + '/recall/resultGauss'
    pathPlotsF1ScoreFinalString = pathPlotsFinalString + '/f1score'
    # pathPlotsF1ScoreGaussFinalString = pathPlotsFinalString + '/f1score/resultGauss'

    effectivePathPlotsAccuracy = os.path.join(pathPlotsAccuracyFinalString)
    # effectivePathPlotsAccuracyGauss = os.path.join(pathPlotsAccuracyGaussFinalString)
    effectivePathPlotsLoss = os.path.join(pathPlotsLossFinalString)
    # effectivePathPlotsLossGauss = os.path.join(pathPlotsLossGaussFinalString)
    effectivePathPlotsPrecision = os.path.join(pathPlotsPrecisionFinalString)
    # effectivePathPlotsPrecisionGauss = os.path.join(pathPlotsPrecisionGaussFinalString)
    effectivePathPlotsRecall = os.path.join(pathPlotsRecallFinalString)
    # effectivePathPlotsRecallGauss = os.path.join(pathPlotsRecallGaussFinalString)
    effectivePathPlotsF1Score = os.path.join(pathPlotsF1ScoreFinalString)
    # effectivePathPlotsF1ScoreGauss = os.path.join(pathPlotsF1ScoreGaussFinalString)

    os.makedirs(effectivePathPlotsAccuracy)
    # os.makedirs(effectivePathPlotsAccuracyGauss)
    os.makedirs(effectivePathPlotsLoss)
    # os.makedirs(effectivePathPlotsLossGauss)
    os.makedirs(effectivePathPlotsPrecision)
    # os.makedirs(effectivePathPlotsPrecisionGauss)
    os.makedirs(effectivePathPlotsRecall)
    # os.makedirs(effectivePathPlotsRecallGauss)
    os.makedirs(effectivePathPlotsF1Score)
    # os.makedirs(effectivePathPlotsF1ScoreGauss)