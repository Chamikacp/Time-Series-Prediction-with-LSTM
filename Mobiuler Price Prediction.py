from pymongo import MongoClient
import numpy
import matplotlib.pyplot as pyPlot
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# To fix a randomSeed
numpy.random.seed(7)

# Create connection with MONGODB
client = MongoClient("mongodb+srv://mobiuler:mobiuler@cluster0-ckl6n.mongodb.net/test?retryWrites=true&w=majority")
mongoDb = client.get_database('Mobiuler')
mCollection = mongoDb.get_collection('price_details')

# List of available phones in database
# availablePhones = ["Apple iPhone Xr"]
availablePhones = []
for doc in mCollection.find():
    name = str(doc['phone_name'])
    availablePhones.append(name)

print(availablePhones)


def prediction():
    # Assigning the dataFrame
    mobiulerDataFrame = pandas.DataFrame(allPrices)
    mobiulerDataFrame.Date = pandas.to_datetime(mobiulerDataFrame.Date)
    mobiulerDataFrame = mobiulerDataFrame.set_index("Date")

    # Creating mobiulerDataset
    mobiulerDataset = mobiulerDataFrame.values

    # Visualize the  price history
    pyPlot.figure(figsize=(16, 8))
    pyPlot.title(phoneModel + ' Price History', fontsize=25)
    pyPlot.plot(mobiulerDataFrame['Price'])
    pyPlot.xlabel('Date', fontsize=18)
    pyPlot.ylabel('Price', fontsize=18)
    pyPlot.show()

    # Normalizing the created mobiulerDataset
    mobiulerScaler = MinMaxScaler(feature_range=(0, 1))
    mobiulerDataset = mobiulerScaler.fit_transform(mobiulerDataset)

    # Spliting the mobiulerDataset into trainData and testData
    trainingDatasetSize = int(len(mobiulerDataset) * 0.67)
    testingDatasetSize = len(mobiulerDataset) - trainingDatasetSize
    trainData = mobiulerDataset[0:trainingDatasetSize, :]
    testData = mobiulerDataset[trainingDatasetSize:len(mobiulerDataset), :]

    # To Convert create a matrix using NumPy
    def createNewDataset(newDataset, backStep):
        dataXArray, dataYArray = [], []
        for i in range(len(newDataset) - backStep):
            a = newDataset[i:(i + backStep), 0]
            dataXArray.append(a)
            dataYArray.append(newDataset[i + backStep, 0])
        return numpy.array(dataXArray), numpy.array(dataYArray)

    # Reshaping the x,y data to t and t+1
    backStep = 1
    trainXData, trainYData = createNewDataset(trainData, backStep)
    testXData, testYData = createNewDataset(testData, backStep)

    # Reshaping the inputData [samples, time steps, features]
    trainXData = numpy.reshape(trainXData, (trainXData.shape[0], 1, trainXData.shape[1]))
    testXData = numpy.reshape(testXData, (testXData.shape[0], 1, testXData.shape[1]))

    # Creating the LSTM model and fit the model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, backStep)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainXData, trainYData, epochs=100, batch_size=1, verbose=2)

    # Predicting Train and Test Data
    trainPrediction = model.predict(trainXData)
    testPrediction = model.predict(testXData)

    # Inverting the predicted data
    trainPrediction = mobiulerScaler.inverse_transform(trainPrediction)
    trainYData = mobiulerScaler.inverse_transform([trainYData])
    testPrediction = mobiulerScaler.inverse_transform(testPrediction)
    testYData = mobiulerScaler.inverse_transform([testYData])

    # Calculating the  RootMeanSquaredError (RMSE)
    phoneTrainingScore = math.sqrt(mean_squared_error(trainYData[0], trainPrediction[:, 0]))
    print('Train Score of a phone: %.2f RMSE' % phoneTrainingScore)
    phoneTestingScore = math.sqrt(mean_squared_error(testYData[0], testPrediction[:, 0]))
    print('Test Score of a phone: %.2f RMSE' % phoneTestingScore)

    # Shifting the trainData for plotting
    trainPredictionPlot = numpy.empty_like(mobiulerDataset)
    trainPredictionPlot[:, :] = numpy.nan
    trainPredictionPlot[backStep:len(trainPrediction) + backStep, :] = trainPrediction

    # Shifting the testData for plotting
    testPredictionPlot = numpy.empty_like(mobiulerDataset)
    testPredictionPlot[:, :] = numpy.nan
    testPredictionPlot[len(trainPrediction) + (backStep * 2) - 1:len(mobiulerDataset) - 1, :] = testPrediction

    # To Plot the available all data,training and tested data
    pyPlot.figure(figsize=(16, 8))
    pyPlot.title(phoneModel + ' Predicted Price', fontsize=25)
    pyPlot.plot(mobiulerScaler.inverse_transform(mobiulerDataset), 'b', label='Original Prices')
    pyPlot.plot(trainPredictionPlot, 'r', label='Trained Prices')
    pyPlot.plot(testPredictionPlot, 'g', label='Predicted Prices')
    pyPlot.legend(loc='upper right')
    pyPlot.xlabel('Number of Days', fontsize=18)
    pyPlot.ylabel('Price', fontsize=18)
    pyPlot.show()

    # To PREDICT FUTURE VALUES
    last_month_price = testPrediction[-1]
    last_month_price_scaled = last_month_price / last_month_price
    next_month_price = model.predict(numpy.reshape(last_month_price_scaled, (1, 1, 1)))
    oldPrice = math.trunc(numpy.ndarray.item(last_month_price))
    newPrice = math.trunc(numpy.ndarray.item(last_month_price * next_month_price))
    print("Last Month Price : ", oldPrice)
    print("Next Month Price : ", newPrice)

    # Updating the predicted price in database
    mobileName = mCollection.find_one({'phone_name': phoneModel})
    if bool(mobileName):
        price_update = {
            'predicted_price': newPrice
        }

        mCollection.update_one({'phone_name': phoneModel}, {'$set': price_update})

        print(phoneModel + " PRICE UPDATED")

        # to clear the array
        allPrices.clear()


allPrices = []

# To find the previous prices of a smartphone
for phoneModel in availablePhones:

    for x in mCollection.find({'phone_name': phoneModel}):
        prices = x['prices']
        for y in prices:
            allPrices.append(y)

        prediction()
