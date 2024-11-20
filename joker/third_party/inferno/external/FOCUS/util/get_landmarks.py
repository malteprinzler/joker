import os
import cv2
import numpy as np
# import tensorflow as tf
from util.preprocess import align_for_lm
# import dlib
import urllib
import bz2
import sys

mean_face = np.loadtxt('util/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])

def save_label(labels, save_path):
    np.savetxt(save_path, labels)

def draw_landmarks(img, landmark, save_name):
    landmark = landmark
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)

    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array([0, 0, 255])
    lm_img = lm_img.astype(np.uint8)

    cv2.imwrite(save_name, lm_img)


# create tensorflow graph for landmark detector
def load_lm_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        lm_sess = tf.Session(graph=graph)

    return lm_sess,img_224,output_lm

def is_filename_img(filepath):
    if '.jpg' in filepath or 'png' in filepath or 'jpeg' in filepath or 'PNG' in filepath or 'bmp' in filepath:
        return True
    else:
        return False


def get_image_namelist(img_path):
    
    names = []
    for dirpath, dirnames, filenames in os.walk(img_path):
        for filename in [f for f in filenames if is_filename_img(f)]:#(f.endswith(".jpg") or f.endswith('.png') or f.endswith('.PNG'))]:
            names.append(os.path.join(dirpath, filename))
    return names

left_eye = [37,38,40,41]
right_eye = [43,44,46,47]
nose = [30]
left_mouth = [48]
right_mouth = [54]

def get_landmarks_main(img_path,save_dir):
    lm_sess,input_op,output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb') # load a tensorflow version 68-landmark detector
    
    
    anno_filepath = os.path.join(save_dir,'list_landmarks.csv')
    
    predictor_path = './Dlib_landmark_detection/model/shape_predictor_68_face_landmarks.dat'  # shape_predictor_68_face_landmarks.dat是进行人脸标定的模型，它是基于HOG特征的，这里是他所在的路径
    if not os.path.exists(predictor_path):
        dlib_model_dir = os.path.split(predictor_path)[0]
        if not os.path.exists(dlib_model_dir):os.makedirs(dlib_model_dir)
        Detector_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        urllib.request.urlretrieve(Detector_URL, './Dlib_landmark_detection/model/shape_predictor_68_face_landmarks.dat.bz2')
        newfile = bz2.decompress( './Dlib_landmark_detection/model/shape_predictor_68_face_landmarks.dat.bz2')
        
    Dlib_detector = dlib.get_frontal_face_detector() #获取人脸分类器
    Dlib_predictor = dlib.shape_predictor(predictor_path)    # 获取人脸检测器


    names = get_image_namelist(img_path)
    print('{} images found.'.format(len(names)))
    detect_68p(Dlib_detector,Dlib_predictor,names,lm_sess,input_op,output_op,save_dir,anno_filepath) # detect landmarks for images
    del Dlib_predictor
    del Dlib_detector
    del lm_sess
    return anno_filepath
def detect_68p(Dlib_detector,Dlib_predictor,names,sess,input_op,output_op,save_path,anno_filepath):
    print('detecting landmarks......')
    
    #if not os.path.isdir(save_path):
        #os.makedirs(save_path)
    
    
    anno_file = open(anno_filepath,mode='w')
    path_not_detected = []
    num_imgs = len(names)
    for i in range(0, len(names)):
        name = names[i]
        #print('%05d' % (i), ' ', name)
        full_image_name = name#os.path.join(img_path, name)
        '''--------------------------------------------------------------------
                                   5 landmarks from dlib
        --------------------------------------------------------------------'''
        
        img= cv2.imread(full_image_name)
        width_img = img.shape[1]
        print("\r",end="")
        print("processing:{:.2f}%: ".format(i/num_imgs *100),"="*(i//int(2*num_imgs/100)),end="")
        sys.stdout.flush()
        
        dets = Dlib_detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
        if len(dets) == 0:
            path_not_detected += [full_image_name]
            continue
        
        for index, face in enumerate(dets):
      
            shape = Dlib_predictor(img, face)  # 寻找人脸的68个标定点 
            points = shape.parts()
            
                
            left_eye_x = 0
            left_eye_y = 0
            for index in left_eye:
                left_eye_x += (points[index].x)/4
                left_eye_y += (points[index].y)/4
            right_eye_x = 0
            right_eye_y = 0
            for index in right_eye:
                right_eye_x += (points[index].x)/4
                right_eye_y += (points[index].y)/4
            nose_xy = points[nose[0]]
            left_mouth_xy = points[left_mouth[0]]
            right_mouth_xy = points[right_mouth[0]]
            string_to_write = str(round(left_eye_x))+'	' + str(round(left_eye_y)) + '\n'+\
                str(round(right_eye_x)) + '	' +str(round(right_eye_y))+ '\n'+\
                            str(round(nose_xy.x)) + '	' +str(round(nose_xy.y))+ '\n'+\
                            str(round(left_mouth_xy.x)) + '	' +str(round(left_mouth_xy.y))+ '\n'+\
                            str(round(right_mouth_xy.x)) + '	' +str(round(right_mouth_xy.y)) + '\n'
            break
        five_points=np.array([left_eye_x,left_eye_y,\
                                right_eye_x,right_eye_y,\
                                nose_xy.x,nose_xy.y,\
                                left_mouth_xy.x,left_mouth_xy.y,\
                                right_mouth_xy.x,right_mouth_xy.y])
        
        
        
        
        '''--------------------------------------------------------------------
                                   68 landmarks from Deep3D
        --------------------------------------------------------------------'''
        
        
        input_img, scale, bbox = align_for_lm(img, five_points) # align for 68 landmark detection 
       
        # if the alignment fails, remove corresponding image from the training list
        if scale == 0:
            path_not_detected += [full_image_name]
            continue

        # detect landmarks
        input_img = np.reshape(
            input_img, [1, 224, 224, 3]).astype(np.float32)
        #cv2.imwrite(full_image_name.replace('/home/li0005/Program/Neural_Representation','/home/li0005/Program/Neural_Representation/valid'\
         #                                   ),input_img[0])
        landmark = sess.run(
            output_op, feed_dict={input_op: input_img})
        

        # transform back to original image coordinate
        landmark = landmark.reshape([68, 2]) + mean_face
        landmark[:, 1] = 223 - landmark[:, 1]
        landmark = landmark / scale
        landmark[:, 0] = landmark[:, 0] + bbox[0]
        landmark[:, 1] = landmark[:, 1] + bbox[1]
        landmark[:, 1] = img.shape[0] - 1 - landmark[:, 1]
        string_to_write = name
        #font = cv2.FONT_HERSHEY_SIMPLEX
        for ip in range(68):
            x=int(landmark[ip][0])
            y=img.shape[0] - 1-int(landmark[ip][1])
            #pt = (x,y)
         #   cv2.circle(img, pt, 1, (255, 0, 0), 2)	
          #  cv2.putText(img, str(ip), pt, font, 0.6, (187, 255, 255), 1, cv2.LINE_AA)

            string_to_write += ','+str(round(x))+',' + str(round(y))
        #cv2.namedWindow('show_tensor', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('show_tensor', cv2.resize(img,(512,512)))
        #cv2.waitKey(0)
        string_to_write +='\n'
        
        anno_file.write(string_to_write)  

        
    print('\n{} faces not detected'.format(len(path_not_detected)))
    
    '''not_found_dir,not_found_ext = os.path.splitext(anno_filepath)
    
    not_found_path = not_found_dir + '_not_detected' + not_found_ext
    not_found_ = open(not_found_path,mode='w')
    not_found_.write('---------------------------------------------------\n')
    not_found_.write('Image dir: ' + img_path + '\n')
    not_found_.write('---------------------------------------------------\n')
    for item_not_found in path_not_detected:
        not_found_.write(item_not_found + '\n')
    not_found_.close()'''
    anno_file.close()
    
    


if __name__ == '__main__':
    img_path = '/home/lcl/Dataset/Adam20221004'
    
    import time
    save_dir = './Results/{}'.format(time.time())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    get_landmarks_main(img_path,save_dir)
                
