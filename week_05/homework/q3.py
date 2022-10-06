import pickle

# The model file
model_file = "./model1.bin"

# Read in model and Dict Vextorizer
with open(model_file, "rb") as fm:
    model = pickle.load(fm)
    
# The DictVectorizer file
dv_file = "./dv.bin"

# Read in model and Dict Vextorizer
with open(dv_file, "rb") as fd:
    dv = pickle.load(fd)
    
    
# The client to score
client = {"reports": 0, 
          "share": 0.001694,
          "expenditure": 0.12, 
          "owner": "yes",}


# Create DictVevtor of client
X = dv.transform([client])


# Predict the client
print(f"Prediction: {round(model.predict_proba(X)[0, 1], 3)}")