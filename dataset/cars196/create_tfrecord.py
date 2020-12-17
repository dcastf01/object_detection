from object_detection.utils import dataset_util
import os
import tensorflow as tf
# @tf.function
def tf_serialize_example(complete_tensor,classlabel):
  '''
  it's necessary a tensor with format dict that have the next keys
  bbox
  filepath
  label
  image
  '''

  #se extrae los puntos de la caja
  box_tensor=complete_tensor["bbox"]
  xmins=[box_tensor[0]]
  xmaxs=[box_tensor[2]]
  ymins=[box_tensor[1]]
  ymaxs=[box_tensor[3]]

  #se extrae el ancho y alto de la imagen
  image=complete_tensor["image"]
  height=image.shape[1]
  width=image.shape[2]

  #se extrae las etiquetas de la imagen
  classes=complete_tensor["label"]

  classes_text=classlabel[classes].encode('utf8')
  
  # classes=1
  # classes_text="car".encode('utf8')
  #se extrae el archivo
  filename=complete_tensor['filename'].numpy()
  path=complete_tensor["filepath"].numpy()

  path_import=os.path.join(str(tf.compat.as_str_any(path)), '{}.jpg'.format(str(tf.compat.as_str_any(filename))))
  image_format=b"jpg"
  
  # id=complete_tensor["id"]
  
  # logging.critical(filename)
  with tf.io.gfile.GFile(path_import, 'rb') as fid:
        encoded_jpg = fid.read()
  tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        # 'image/source_id': dataset_util.bytes_feature(id),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),

        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_feature(classes_text),
        'image/object/class/label': dataset_util.int64_feature(classes),
    }))
  
  # tf_string = tf.py_function(
  #   serialize_exampleba,
  #   f0,  # pass these args to the above function.
  #   tf.string)      # the return type is `tf.string`.
  return tf_example

def write_TFrecord(ds,classlabel,path_annotation="/content/workspace/annotations/",subsetname="train"):
  for features in ds:
    tf_example=tf_serialize_example(features,classlabel)
    writer = tf.io.TFRecordWriter(path_annotation+"/"+subsetname+".record")
    writer.write(tf_example.SerializeToString())
  writer.close()
