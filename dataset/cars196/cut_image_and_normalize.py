import tensorflow as tf
class augmentation:
    
    def __init__(self,PROBABLITY_THRESHOLD,PERCENT_INCREMENTED_IN_JITTER,AUGMENTED,BATCH_SIZE,SIZE):
        PROBABLITY_THRESHOLD = 0.4 #@param {type:"slider", min:0, max:1, step:0.1}
        #PERCENT_INCREMENTED_IN_JITTER value recomended 0.11
        PERCENT_INCREMENTED_IN_JITTER= 0.11 #@param {type:"slider", min:0, max:0.25, step:0.01}
        AUGMENTED = True #@param {type:"boolean"}
        BATCH_SIZE = 32 #normalmente en 32 
        SIZE=224#@param {type:"integer"}
        IMAGE_SIZE=[SIZE,SIZE]
        IMG_HEIGHT = SIZE 
        IMG_WIDTH = SIZE

    @tf.function
    def crop_image_and_normalize_and_augmented(self,tensor,augmented=True):
        def crop_image(img,bbox):
        #las cajas siguen esta convención ymin, xmin, ymax, xmax
            shape=img.get_shape()
            width=shape[0]
            height=shape[1]

            # ymin, xmin, ymax, xmax=bbox
            ymin=tf.cast(tf.multiply(bbox[0],height),tf.int32)
            ymax=tf.cast(tf.multiply(bbox[2],height),tf.int32)
            xmin=tf.cast(tf.multiply(bbox[1],width),tf.int32)
            xmax=tf.cast(tf.multiply(bbox[3],width),tf.int32)

            result= tf.image.crop_to_bounding_box(img, xmin, ymin, 
                                                xmax-xmin, ymax-ymin)
            return result

        def resize(img,height,width):

            img=tf.image.resize(img,[height,width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
            return img

        def normalize(img):
            
            img=tf.keras.applications.imagenet_utils.preprocess_input(img,mode="tf")
        
            return img

        def random_jitter(img,height,width):
            
            incremented=PERCENT_INCREMENTED_IN_JITTER
            height_incremented=int(height*(1+incremented))
            width_incremented=int(width*(1+incremented))
            img = resize(img,height_incremented,width_incremented)
            
            img = tf.image.random_crop(img, size=[height,width,3])
            
            return img
        def flip_image(img: tf.Tensor):
            """Flip augmentation

            Args:
                img: Image

            Returns:
                Augmented image
            """
            if tf.random.uniform(()) > PROBABLITY_THRESHOLD :
            
            img=tf.image.flip_left_right(img)
            return img
        
        def color(img: tf.Tensor) -> tf.Tensor:
            """Color augmentation

            Args:
                img: Image

            Returns:
                Augmented image
            """
            if tf.random.uniform(()) > PROBABLITY_THRESHOLD :
            img = tf.image.random_hue(img, 0.08)
            if tf.random.uniform(()) > PROBABLITY_THRESHOLD :
            img = tf.image.random_saturation(img, 0.6, 1.6)
            if tf.random.uniform(()) > PROBABLITY_THRESHOLD :
            img = tf.image.random_brightness(img, 0.05)
            if tf.random.uniform(()) > PROBABLITY_THRESHOLD :
            img = tf.image.random_contrast(img, 0.7, 1.3)
            
            return img

        def encode_label(label):
            return tf.one_hot(label,NUM_CLASSES)


        img=tensor["image"]
        img=tf.ensure_shape(tf.cast(tf.py_function(crop_image, [img,tensor["bbox"]], [tf.uint8]),tf.float32),[None,None, None, 3])
        
        img=tf.squeeze(img,[0])
        img=resize(img,IMG_HEIGHT,IMG_WIDTH)

        if augmented:
            img=random_jitter(img,IMG_HEIGHT,IMG_WIDTH)
            img=flip_image(img)
            img=color(img)
            

        img=normalize(img)
        img=tf.clip_by_value(img, -1, 1)

        label=encode_label(tensor["label"])
        return img,label

        @tf.function
        def load_train_image(self,tensor):
            return crop_image_and_normalize_and_augmented(tensor,AUGMENTED)
        @tf.function
        def load_test_image(self,tensor):

            return crop_image_and_normalize_and_augmented(tensor)