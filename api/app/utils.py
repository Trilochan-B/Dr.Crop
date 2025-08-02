import pickle
import os
import sys
import cv2
import numpy as np
import google.generativeai as genai
from tensorflow.keras.models import load_model
from exception import CustonmException


class preprocess():
    def load_object(self,path :str):
        try:
            with open(path,'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise CustonmException(e,sys)
    def crop_preprocess(self,data):
        try:
            preprocessor = self.load_object('res/crop/Preprocessor.pkl')
            data_preprocessed = preprocessor.transform([data])
            return data_preprocessed
        except Exception as e:
            raise CustonmException(e,sys)
    
    def predict_crop(self,data):
        try:
            model = self.load_object('res/crop/crop_model.pkl')
            data_preprocessed = self.crop_preprocess(data)
            prediction = model.predict(data_preprocessed)
            return prediction[0]
        except Exception as e:
            raise CustonmException(e,sys)
    
    def load_image(self,image_path):
        try:
            path = "D:\BCA\Test_data"
            image = cv2.imread(os.path.join(path,image_path))
            return np.array(image)
        except Exception as e:
            raise CustonmException(e,sys)
    
    def image_preprocess(self,image_path):
        try:
            image = self.load_image(image_path)
            image_resize = cv2.resize(image, (252,252))
            image_scaled = image_resize/255
            image = image_scaled[np.newaxis,...]
            return image
        except Exception as e:
            raise CustonmException(e,sys)
    
    def predict_disease(self,data):
        try:
            label = self.load_object(f"res/disease/{data['crop']}Labels.pkl")
            image = self.image_preprocess(data['image'])
            model = load_model(f"res/disease/{data['crop']}CNN.h5")
            prediction = model.predict(image)[0]
            index = np.argmax(prediction)
            return label[index],prediction[index]
        except Exception as e:
            raise CustonmException(e,sys)
        
    
class describe():
    def __init__(self):
        api_key= 'api_key'
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name = 'gemini-2.0-flash')
    def crop(self,data):
        soil = f"Nitrogen = {data[0]}kg/ha, Phosphorus = {data[1]}kg/ha, Kalium = {data[2]}kg/ha, temperature = {data[3]}celsius, humidity = {data[4]}%, pH = {data[5]},rainfall = {data[6]}mm"
        crop = data[7]
        task = f"""You are a crop specialist assistant. User's soil component details is {soil} recommended crop is {crop}. According to user's soil analysis a crop is suggested to the farmer. The soil data and the suggested crop name is provided you, write a short note on how the soil is perfect for the crop around 100 words."""

        responce = self.model.generate_content(task)
        return responce.text.split('**')[-1]
    def preferance(self,data):
        soil = f"Nitrogen = {data[0]}kg/ha, Phosphorus = {data[1]}kg/ha, Kalium = {data[2]}kg/ha, temperature = {data[3]}celsius, humidity = {data[4]}%, pH = {data[5]},rainfall = {data[6]}mm"
        crop = data[7]
        task = f"""You are a crop specialist assistant. User's soil component details is {soil}. User wants to grow {crop}. According to user's soil quality, write a short note on how user prepare the soil for the crop that he wants to grow."""

        responce = self.model.generate_content(task)
        return responce.text.split('**')[-1]

