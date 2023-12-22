import os
import cv2
import pandas as pd
import csv
from deepface import DeepFace


#Create the starting variables
imagefolder = 'faceimages'
imagearray = []
results = []

# Import the images from the file
def importImagestoArray():
    for filename in os.listdir(imagefolder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(imagefolder, filename)
            image = cv2.imread(image_path)
            imagearray.append(image)

# Validate the images throug DeepFace
def validateimages(array):
    for item in array:
        verify = DeepFace.analyze(item, actions =['gender', 'age', 'race'], enforce_detection=False)
        gender = verify[0]['gender']
        ethnicity = verify[0]['race']
        age = verify[0]['age']
        results.append({'Gender': gender, 'Race': ethnicity, 'Age': age})


# Main function that runs the other functions.
def main():
   importImagestoArray()
   validateimages(imagearray)
   df_results = pd.DataFrame(results)
   df_results['Gender'] = df_results['Gender'].apply(lambda x:max(x, key=lambda k: x[k]))
   df_results['Race'] = df_results['Race'].apply(lambda x:max(x, key=lambda k: x[k]))
   df_results.to_csv('Results.cvs')
   print(df_results)
  
   


main()