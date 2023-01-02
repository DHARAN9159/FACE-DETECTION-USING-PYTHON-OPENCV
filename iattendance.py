
import os
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import face_recognition
from flask import Flask, render_template, Response, request, url_for, redirect

app = Flask(__name__)


def create_students_data():
    if 'StudentData.csv' not in os.listdir():
        data = pd.DataFrame(data=None, columns=
            ['reg_no', 'first_name', 'second_name', 'address', 'phone_no', 'alt_phone_no'])
        data.to_csv('StudentData.csv', index=False)
    else:
        pass

def create_attendance_sheet(subjects):
    for subject in subjects:
        if f'{subject}.csv' not in os.listdir('AttendanceData'):
            data = pd.DataFrame(data=None,columns=['reg_no', 'name'])
            data.to_csv(f'AttendanceData/{subject}.csv', index=False)

images = []
classNames = []
def get_photo():
	path = 'StudentsImage'
	myList = os.listdir(path)
	for cl in myList:
	    curImg = cv2.imread(f'{path}/{cl}')
	    images.append(curImg)
	    classNames.append(os.path.splitext(cl)[0])
	print(classNames)
 
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def generate_frames(encodeListKnown, subject, today_date, current_time):
    while True:
        camera=cv2.VideoCapture(0)
        ## read the camera frame
        success,img=camera.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    registerNumber = classNames[matchIndex]
                    print(registerNumber)

                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,registerNumber,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                    data = pd.read_csv(f'AttendanceData/{subject}.csv')
                    print(data['reg_no'])
                    idx = data[data['reg_no'] == str(registerNumber)].index[0]

                    try:
                        if data[today_date][idx] == '-':
                            data[today_date][idx] = data[today_date][idx].replace(data[today_date][idx], 
                                f"P ({current_time})")

                            data.to_csv(f'AttendanceData/{subject}.csv', index=False)
                    except:
                        data[str(today_date)] = '-'
                        if data[today_date][idx] == '-':
                            data[today_date][idx] = data[today_date][idx].replace(data[today_date][idx], 
                                f"P ({current_time})") 

                            data.to_csv(f'AttendanceData/{subject}.csv', index=False)

            ret,buffer=cv2.imencode('.jpg',img)
            img=buffer.tobytes()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


subject_list = ['Mathematics', 'Science', 'Social']

create_students_data()
create_attendance_sheet(subject_list)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('login.html', title='Login')

    else:
        password = request.form['password']
        if password == 'fuckingjb':
            return render_template('home.html')
        else:
            return render_template('login.html', title='Login')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/take_attendance', methods=['POST', 'GET'])
def take_attendance():

    if request.method == 'POST':

        sub = request.form['subject']
        for subject in subject_list:
            if sub == subject:
                get_photo()
                encodeListKnown = find_encodings(images)

                today_date = pd.to_datetime('today').strftime('%d:%m:%Y, %I:%M%p')
                current_time = pd.to_datetime('today').strftime("%I:%M%p")

                return Response(generate_frames(encodeListKnown, subject, today_date, current_time), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        return render_template('take_attendance.html', subject_list=subject_list)
    

@app.route('/add_students', methods=['POST', 'GET'])
def add_students():
    if request.method == 'POST':
        reg_no = request.form['regno']
        first_name = request.form['fname']
        second_name = request.form['sname']
        address = request.form['adress']
        phone_no = request.form['phno']
        alt_phone_no = request.form['phno_2']
        student_image = request.files['photo']

        if reg_no and first_name and address and phone_no and student_image:
            student_data = pd.read_csv('StudentData.csv')
            reg_num_list = [i for i in student_data.reg_no]

            if str(reg_no) not in reg_num_list:
                student_photo = Image.open(student_image)  
                student_photo = student_photo.save('StudentsImage/{}.jpeg'.format(reg_no))
                student_data = student_data.append({
                                                    'reg_no': str(reg_no), 
                                                    'first_name': str(first_name),
                                                    'second_name': str(second_name),
                                                    'address': str(address),
                                                    'phone_no': str(phone_no),
                                                    'alt_phone_no': str(alt_phone_no)
                                                    }, ignore_index=True)

                student_data.to_csv('StudentData.csv', index=False)

                for subject in subject_list:
                    sub_data = pd.read_csv(f'AttendanceData/{subject}.csv')
                    sub_data = sub_data.append({'reg_no': reg_no, 
                                                'name':first_name}, ignore_index=True)

                    sub_data.to_csv(f'AttendanceData/{subject}.csv', index=False)

            return render_template('add_students.html')

        else:
            return render_template('add_students.html')
        

    else:
        return render_template('add_students.html')

@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    if request.method == 'POST':
        sub = request.form['subject']

        for subject in subject_list:
            if sub == subject:
                attendance_data = pd.read_csv(f'AttendanceData/{subject}.csv')
                headings = attendance_data.columns.values
                data = attendance_data.values
                return render_template('view_attendance.html', headings=headings, data=data, 
                                        subject_list=subject_list, selected_sub=sub)

    else:
        return render_template('view_attendance.html', subject_list=subject_list)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)

