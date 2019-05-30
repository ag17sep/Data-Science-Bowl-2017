import os

MAIN_DIRECTORY = os.path.dirname(os.getcwd()) #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge"
TRAINING_SET = os.path.join(os.path.dirname(os.getcwd()),"Training Set") #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\Training Set"
TEST_SET = os.path.join(os.path.dirname(os.getcwd()),"Test Set") #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\Test Set"
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),"data") #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\data"
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.getcwd()),"model_weights") #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\model_weights"
CSV_TARGET = os.path.join(DATA_PATH,'test') #r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\data\test"
