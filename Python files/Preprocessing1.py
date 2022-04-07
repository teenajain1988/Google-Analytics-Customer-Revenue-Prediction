import pandas as pd
import numpy as np
import json
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import logging



logging.basicConfig(
    filename='logging_preprocessing_file1.txt',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode="a"
    
)

logging.warning("Warning message while importing the module")



cluster = Cluster(
    contact_points=['127.0.0.1'], 
    auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
)
session = cluster.connect()
session.set_keyspace('custom')
session.row_factory = pandas_factory
session.default_fetch_size = 10000000 #needed for large queries, otherwise driver will do pagination. Default is 50000.


sql_query1 = "SELECT * FROM {}.{};".format(CASSANDRA_DB, CASSANDRA_TRAIN_TABLE)
sql_query2 = "SELECT * FROM {}.{};".format(CASSANDRA_DB, CASSANDRA_TEST_TABLE)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

for row in session.execute(sql_query1):
    train_df = train_df.append(pd.DataFrame(row, index=[0]))

for row in session.execute(sql_query2):
    test_df = test_df.append(pd.DataFrame(row, index=[0]))
    
    
logging.info("Date appended in pandas daatframe from casandra database")

train_df.to_pickle("train_data")
test_df.to_pickle("test_df")

def change_customDimensions(x):
    if x == "[]":
        return np.nan
    else:
        return x[26:-3]
train_df["customDimensions"]=train_df["customDimensions"].apply(change_customDimensions)
train_df["customDimensions"]=test_df["customDimensions"].apply(change_customDimensions)



# Converting Json column into Pandas DataFrame

def convert_json_col(data):
    columns=["device","geoNetwork","totals","trafficSource"]
    for col in columns:
        data[col]=data[col].map(lambda x:json.loads(x))
        df2=pd.json_normalize(data[col])
        data.columns=[f"{col}.{subcol}" for subcol in data.columns]
        data=pd.concat([data,df2],axis=1)
        
train_df=convert_json_col(train_df)
test_df=convert_json_col(test_df)
logging.info("Converted JSON column into Pandas Dataframe")



# "hit" column does not give any information so just delete it.
test_df=test_df.drop(["device","geoNetwork","totals","trafficSource","hits"],axis=1)
train_df=train_df.drop(["device","geoNetwork","totals","trafficSource","hits"],axis=1)
logging.info("Features which are not informative have been dropped")



def drop_unusual_column(data):
    ### These columns does not contain any informations so drop these.
    col_to_drop=["device.browserSize","device.browserVersion","device.flashVersion","device.language",
                "device.mobileDeviceBranding","device.mobileDeviceInfo","device.mobileDeviceMarketingName",
                "totals.visits","geoNetwork.cityId","geoNetwork.latitude","geoNetwork.longitude",
                "geoNetwork.networkLocation","trafficSource.adwordsClickInfo.criteriaParameters","socialEngagementType"]
    train.drop(col_to_drop,inplace=True,axis=1)
    
train_df=drop_unusual_column(train_df)
test_df=drop_unusual_column(test_df)

logging.info("Deleted columns which does not give any information to our model")



train_test_df=pd.concat([train,test],axis=0)

logging.info("Merged the train and test dataframe")

