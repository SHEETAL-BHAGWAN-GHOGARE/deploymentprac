from flask import Flask,render_template,request,jsonify
import mysql.connector


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student_data', methods = ['POST'])
def student_data():
    s_name = request.form['student_name']
    s_roll_no = request.form['student_roll_no']
    s_sub1 = request.form['student_sub1']
    s_sub2 = request.form['student_sub2']
    s_sub3 = request.form['student_sub3']

    print(f"Student Name = {s_name}")

    conn = mysql.connector.connect( host= 'localhost',
                                    database = 'may7',
                                    user= 'root',
                                    password = 'admin')

    cursor = conn.cursor()

    query = "INSERT INTO student(name,roll_no,sub1,sub2,sub3) VALUES(%s,%s,%s,%s,%s)"
    data = (s_name,s_roll_no,s_sub1,s_sub2,s_sub3)

    cursor.execute(query,data)

    conn.commit()
    conn.close()

    return "Student Data recived"

if __name__== "__main__":
    app.run(debug=True)