import cv2
from apporchid.common.logger import logger

def get_image(input_dir , file):
    
    try:
        image = cv2.imread(input_dir + file)
    
    except Exception as e:
        logger.debug('Unable to extract files')
        logger.exception(e)
    
    return image
