from keras.layers import Input, Dense, Activation, LeakyReLU, Conv2D, MaxPooling2D, Reshape
from keras.models import Model, Sequential, load_model
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import sys

#Load the fashion mnist data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) #has to add a dimension for channels to work with stuff
x_test = x_test.reshape(-1,28,28,1)

#Trasform lables to be one hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#variable for inputs to the models
inputs = Input(shape=(28,28,1))

if("retrain-teacher" in sys.argv):
    #Define the teacher model
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = Conv2D(32, (3,3), activation='relu')(x)
    hinter = Model(inputs=inputs, outputs=x)
    hinter.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    x = hinter(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Reshape([1024])(x)
    x = Dense(128, activation='relu')(x)
    logits = Dense(10)(x)
    teacherLogits = Model(inputs=inputs, outputs=logits)
    teacherLogits.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    logits = teacherLogits(inputs)
    predictions = Activation('softmax')(logits)
    teacher = Model(inputs=inputs, outputs=predictions)

    print(teacher.summary())

    teacher.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    teacher.fit(x_train, Y_train, batch_size=128, epochs=1)
    print(teacher.evaluate(x_test, Y_test))

    teacher.save('kerasSaved/teacher.h5')
    teacherLogits.save('kerasSaved/teacherLogits.h5')
    hinter.save('kerasSaved/hinter.h5')
else:
    teacher = load_model('kerasSaved/teacher.h5')
    teacherLogits = load_model('kerasSaved/teacherLogits.h5')
    hinter = load_model('kerasSaved/hinter.h5')
    print("should have loaded models")

#Define the student model
x = Conv2D(16, (3,3), activation='relu')(inputs)
x = Conv2D(16, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(32, (3,3), activation='relu')(x)
x = Conv2D(32, (3,3), activation='relu')(x)

guided = Model(inputs=inputs, outputs=x)
guided.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                metrics=['accuracy'])
x = guided(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(48, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Reshape([48])(x)
x = Dense(128)(x)
logits = Dense(10)(x)
studentLogits = Model(inputs=inputs, outputs=logits)
studentLogits.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                metrics=['accuracy'])
logits = studentLogits(inputs)
predictions = Activation('softmax')(logits)

student = Model(inputs=inputs, outputs=predictions)
student.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print(student.summary())


if "dark_knowledge" in sys.argv:
    soft_labels = teacherLogits.predict(x_train)
    student.fit(x_train, soft_labels, batch_size=128, epochs=1)
else:
    student.fit(x_train, Y_train, batch_size=128, epochs=1)
print(student.evaluate(x_test, Y_test))


