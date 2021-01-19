import tensorflow as tf

##no finish

with strategy.scope(): # this line is all that is needed to run on TPU (or multi-GPU, ...)

  bnmomemtum=0.9
  def fire(x, squeeze, expand):
   
    if con_batch:
      x=tf.keras.layers.BatchNormalization()(x)
    if con_dropout:
      x=tf.keras.layers.Dropout(value_dropout)(x)
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    
    y1 = tf.keras.layers.Conv2D(filters=expand, kernel_size=1, activation='relu', padding='same')(y)
 
    y3 = tf.keras.layers.Conv2D(filters=expand, kernel_size=3, activation='relu', padding='same')(y)
   
   
    return tf.keras.layers.concatenate([y1, y3])

  def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)
    
  def fire_with_bypass(x, squeeze, expand):
    
    if con_batch:
      x=tf.keras.layers.BatchNormalization()(x)
    #if con_dropout:
     # x=tf.keras.layers.Dropout(value_dropout)(x)
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)

    y1 = tf.keras.layers.Conv2D(filters=expand, kernel_size=1, activation='relu', padding='same')(y)
 
    y3 = tf.keras.layers.Conv2D(filters=expand, kernel_size=3, activation='relu', padding='same')(y)
   
    y5=tf.keras.layers.concatenate([y1, y3])
    return tf.keras.layers.add([x, y5])

  def fire_module_with_bypass(squeeze, expand):
    return lambda x: fire_with_bypass(x, squeeze, expand)
 
  x = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3]) # input is 192x192 pixels RGB

  y = tf.keras.layers.Conv2D(kernel_size=3, filters=95,strides=2, use_bias=True, activation='relu')(x)
  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
  y = fire_module(16, 64)(y)
  y = fire_module_with_bypass(16, 64)(y)
  y = fire_module(32, 128)(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
  y = fire_module_with_bypass(32, 128)(y)
  y = fire_module(48, 192)(y)
  y = fire_module_with_bypass(48, 192)(y)
  y = fire_module(64, 256)(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
  y = fire_module_with_bypass(64, 256)(y)
  
  if con_batch:
    y=tf.keras.layers.BatchNormalization()(y)
  if con_dropout:
      y=tf.keras.layers.Dropout(value_dropout)(y)
  y = tf.keras.layers.Conv2D(kernel_size=3, filters=NUM_CLASS,strides=2, use_bias=True, activation='relu')(y) #en el modelo el numero de filtros es igual al número de clases
  y = tf.keras.layers.GlobalAveragePooling2D()(y)
  y = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(y) 




  model = tf.keras.Model(x, y)
  
  # initiate RMSprop optimizer
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005, decay=1e-6)

  METRICS=[ 
      
      tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
      tf.keras.metrics.Accuracy(name="accuracy_binary"),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc',num_thresholds=500),#cambiar el auc
      tf.keras.metrics.TopKCategoricalAccuracy( k=3, name='top_3_categorical_accuracy'),
      tf.keras.metrics.TopKCategoricalAccuracy( k=5, name='top_5_categorical_accuracy'),
      tf.keras.metrics.TopKCategoricalAccuracy( k=10, name='top_10_categorical_accuracy')
        ]
    # Let's train the model using RMSprop
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=METRICS)