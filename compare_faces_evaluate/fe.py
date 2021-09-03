import dlib
'''
import _dlib_pybind11.cuda as cuda # <module '_dlib_pybind11.cuda'>
import _dlib_pybind11.image_dataset_metadata as image_dataset_metadata # <module '_dlib_pybind11.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins
'''
import numpy as np


'''
#dlib
class cnn_face_detection_model_v1(__pybind11_builtins.pybind11_object):
    """ This object detects human faces in an image.  The constructor loads the face detection model from a file. You can download a pre-trained model from http://dlib.net/files/mmod_human_face_detector.dat.bz2. """
    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: _dlib_pybind11.cnn_face_detection_model_v1, imgs: list, upsample_num_times: int=0, batch_size: int=128) -> std::__1::vector<std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> >, std::__1::allocator<std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> > > >
        
        takes a list of images as input returning a 2d list of mmod rectangles
        
        2. __call__(self: _dlib_pybind11.cnn_face_detection_model_v1, img: array, upsample_num_times: int=0) -> std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> >
        
        Find faces in an image using a deep learning model.
                  - Upsamples the image upsample_num_times before running the face 
                    detector.
        """
        pass

    def __init__(self, filename): # real signature unknown; restored from __doc__
        """ __init__(self: _dlib_pybind11.cnn_face_detection_model_v1, filename: str) -> None """
        pass
'''

'''
#dlib
def get_frontal_face_detector(): # real signature unknown; restored from __doc__
    """
    get_frontal_face_detector() -> dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6u>, dlib::default_fhog_feature_extractor> >
    
    Returns the default face detector
    """
    pass
'''

# face_recognition
'''
# 'init.py' in 'face_recognition models'
from pkg_resources import resource_filename

def pose_predictor_model_location():
    return resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return resource_filename(__name__, "models/mmod_human_face_detector.dat")
'''

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()
cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()

cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model) #dlib
face_detector = dlib.get_frontal_face_detector() #dlib



# face_recognition
def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)



'''
class rectangle(__pybind11_builtins.pybind11_object):
    """ This object represents a rectangular area of an image. """
    def area(self): # real signature unknown; restored from __doc__
        """ area(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def bl_corner(self): # real signature unknown; restored from __doc__
        """
        bl_corner(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point
        
        Returns the bottom left corner of the rectangle.
        """
        pass

    def bottom(self): # real signature unknown; restored from __doc__
        """ bottom(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def br_corner(self): # real signature unknown; restored from __doc__
        """
        br_corner(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point
        
        Returns the bottom right corner of the rectangle.
        """
        pass

    def center(self): # real signature unknown; restored from __doc__
        """ center(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point """
        pass

    def contains(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        contains(*args, **kwargs)
        Overloaded function.
        
        1. contains(self: _dlib_pybind11.rectangle, point: _dlib_pybind11.point) -> bool
        
        2. contains(self: _dlib_pybind11.rectangle, point: _dlib_pybind11.dpoint) -> bool
        
        3. contains(self: _dlib_pybind11.rectangle, x: int, y: int) -> bool
        
        4. contains(self: _dlib_pybind11.rectangle, rectangle: _dlib_pybind11.rectangle) -> bool
        """
        pass

    def dcenter(self): # real signature unknown; restored from __doc__
        """ dcenter(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point """
        pass

    def height(self): # real signature unknown; restored from __doc__
        """ height(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def intersect(self, rectangle): # real signature unknown; restored from __doc__
        """ intersect(self: _dlib_pybind11.rectangle, rectangle: _dlib_pybind11.rectangle) -> _dlib_pybind11.rectangle """
        pass

    def is_empty(self): # real signature unknown; restored from __doc__
        """ is_empty(self: _dlib_pybind11.rectangle) -> bool """
        return False

    def left(self): # real signature unknown; restored from __doc__
        """ left(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def right(self): # real signature unknown; restored from __doc__
        """ right(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def tl_corner(self): # real signature unknown; restored from __doc__
        """
        tl_corner(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point
        
        Returns the top left corner of the rectangle.
        """
        pass

    def top(self): # real signature unknown; restored from __doc__
        """ top(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def tr_corner(self): # real signature unknown; restored from __doc__
        """
        tr_corner(self: _dlib_pybind11.rectangle) -> _dlib_pybind11.point
        
        Returns the top right corner of the rectangle.
        """
        pass

    def width(self): # real signature unknown; restored from __doc__
        """ width(self: _dlib_pybind11.rectangle) -> int """
        return 0

    def __add__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __add__(*args, **kwargs)
        Overloaded function.
        
        1. __add__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.point) -> _dlib_pybind11.rectangle
        
        2. __add__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.rectangle) -> _dlib_pybind11.rectangle
        """
        pass

    def __eq__(self, arg0): # real signature unknown; restored from __doc__
        """ __eq__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.rectangle) -> bool """
        return False

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: _dlib_pybind11.rectangle) -> tuple """
        return ()

    def __iadd__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __iadd__(*args, **kwargs)
        Overloaded function.
        
        1. __iadd__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.point) -> _dlib_pybind11.rectangle
        
        2. __iadd__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.rectangle) -> _dlib_pybind11.rectangle
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: _dlib_pybind11.rectangle, left: int, top: int, right: int, bottom: int) -> None
        
        2. __init__(self: _dlib_pybind11.rectangle, rect: dlib::drectangle) -> None
        
        3. __init__(self: _dlib_pybind11.rectangle, rect: _dlib_pybind11.rectangle) -> None
        
        4. __init__(self: _dlib_pybind11.rectangle) -> None
        """
        pass

    def __ne__(self, arg0): # real signature unknown; restored from __doc__
        """ __ne__(self: _dlib_pybind11.rectangle, arg0: _dlib_pybind11.rectangle) -> bool """
        return False

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: _dlib_pybind11.rectangle) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: _dlib_pybind11.rectangle, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: _dlib_pybind11.rectangle) -> str """
        return ""
'''

# face_recognition
def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2]) #dlib



'''
class shape_predictor(__pybind11_builtins.pybind11_object):
    """ This object is a tool that takes in an image region containing some object and outputs a set of point locations that define the pose of the object. The classic example of this is human face pose prediction, where you take an image of a human face as input and are expected to identify the locations of important facial landmarks such as the corners of the mouth and eyes, tip of the nose, and so forth. """
    def save(self, predictor_output_filename): # real signature unknown; restored from __doc__
        """
        save(self: _dlib_pybind11.shape_predictor, predictor_output_filename: str) -> None
        
        Save a shape_predictor to the provided path.
        """
        pass

    def __call__(self, image, box): # real signature unknown; restored from __doc__
        """
        __call__(self: _dlib_pybind11.shape_predictor, image: array, box: _dlib_pybind11.rectangle) -> _dlib_pybind11.full_object_detection
        
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - box is the bounding box to begin the shape prediction inside. 
        ensures 
            - This function runs the shape predictor on the input image and returns 
              a single full_object_detection.
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: _dlib_pybind11.shape_predictor) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: _dlib_pybind11.shape_predictor) -> None
        
        2. __init__(self: _dlib_pybind11.shape_predictor, arg0: str) -> None
        
        Loads a shape_predictor from a file that contains the output of the 
        train_shape_predictor() routine.
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: _dlib_pybind11.shape_predictor, arg0: tuple) -> None """
        pass
'''

# face_recognition
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()

pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model) #dlib
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model) #dlib

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations] #face_recognition

    pose_predictor = pose_predictor_68_point #face_recognition

    if model == "small":
        pose_predictor = pose_predictor_5_point #face_recognition

    return [pose_predictor(face_image, face_location) for face_location in face_locations] #face_recognition



'''
class face_recognition_model_v1(__pybind11_builtins.pybind11_object):
    """ This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 """
    def compute_face_descriptor(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        compute_face_descriptor(*args, **kwargs)
        Overloaded function.
        
        1. compute_face_descriptor(self: _dlib_pybind11.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], face: _dlib_pybind11.full_object_detection, num_jitters: int=0, padding: float=0.25) -> _dlib_pybind11.vector
        
        Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        2. compute_face_descriptor(self: _dlib_pybind11.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], num_jitters: int=0) -> _dlib_pybind11.vector
        
        Takes an aligned face image of size 150x150 and converts it into a 128D face descriptor.Note that the alignment should be done in the same way dlib.get_face_chip does it.If num_jitters>1 then image will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. 
        
        3. compute_face_descriptor(self: _dlib_pybind11.face_recognition_model_v1, img: numpy.ndarray[(rows,cols,3),uint8], faces: _dlib_pybind11.full_object_detections, num_jitters: int=0, padding: float=0.25) -> _dlib_pybind11.vectors
        
        Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        4. compute_face_descriptor(self: _dlib_pybind11.face_recognition_model_v1, batch_img: List[numpy.ndarray[(rows,cols,3),uint8]], batch_faces: List[_dlib_pybind11.full_object_detections], num_jitters: int=0, padding: float=0.25) -> _dlib_pybind11.vectorss
        
        Takes an array of images and an array of arrays of full_object_detections. `batch_faces[i]` must be an array of full_object_detections corresponding to the image `batch_img[i]`, referencing faces in that image. Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        
        5. compute_face_descriptor(self: _dlib_pybind11.face_recognition_model_v1, batch_img: List[numpy.ndarray[(rows,cols,3),uint8]], num_jitters: int=0) -> _dlib_pybind11.vectors
        
        Takes an array of aligned images of faces of size 150_x_150.Note that the alignment should be done in the same way dlib.get_face_chip does it.Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor.
        """
        pass

    def __init__(self, arg0): # real signature unknown; restored from __doc__
        """ __init__(self: _dlib_pybind11.face_recognition_model_v1, arg0: str) -> None """
        pass
'''

# face_recognition
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model) #dlib
def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]