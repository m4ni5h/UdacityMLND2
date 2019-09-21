The Data Exploration and XGBOOST analysis are done in Jupyter notebook(1_DataExploration.ipynb, 2_XGBOOST.ipynb) which can accessed from https://github.com/m4ni5h/UdacityMLND2/tree/master/CapstoneProject/ 
Steps to Reproduce Similar Results(Final Solution):
1. Download the data from [WSDM - KKBox's Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/overview)
2. Download the code checked in the GitHub Repository https://github.com/m4ni5h/UdacityMLND2/tree/master/CapstoneProject/
3. Copy the data csv files(members.csv, song_extra_info.csv, songs.csv, test.csv and train.csv) to /CapstoneProject/Project/input/training/source_data
4. Run the run.sh located in training/script folder, this will do feature extraction and create final csv files(members_add.csv, members_gbdt.csv, members_nn.csv, songs_gbdt.csv, songs_nn.csv, test.csv, test_add.csv, train.csv, train_part.csv, train_part_add.csv) in training folder.
5. Finally run nn_training.py with nn_record.csv in the same folder.