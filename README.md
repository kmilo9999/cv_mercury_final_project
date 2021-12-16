### How to:
1. clean the dataset: navigate to code directory, run `python dataset_cleanup.py`, this should automatically download egohands dataset and split the train and test data, and images and csv files are generated under images folder in the code directory
2. Move train.csv and test.csv files (in images/train and images/test respectively) into a new "data" folder under code directory 
The file structure is  like this before generating tfrecords:
-data/
--test_labels.csv
--train_labels.csv
-images/
--test/
---testingimages.jpg
--train/
---testingimages.jpg
3. under code directory, 
    - run `python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/` to generate training data tfrecords.
    - run `python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test` to generate testing data records.

The final file structure should look like this:

![file structure](/file_structure.png?raw=true "File Structure")

### How to: 
4. Run the hand detection interactive game using the weights generated from our trained model:
Navigate to code directory, run `python main.py`