from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(path,targetSize):
    datagen = ImageDataGenerator(rescale = 1./255)
    images = datagen.flow_from_directory(directory = path, target_size=targetSize, color_mode = 'rgb', class_mode = None, seed = 301)
    return images
