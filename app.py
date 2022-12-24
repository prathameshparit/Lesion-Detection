import os
import csv
import webbrowser
from flask import Flask, render_template, request
# from dental import pred_data
from plot import comparitive_plot
from PIL import Image
from preprocess import processing, hsv, filteration
import matplotlib
import tensorflow as tf
import pickle
import pandas as pd
from predictions import pred_and_plot, pred_ml_model

matplotlib.use('Agg')

# loading the saved models
CNN_model = tf.keras.models.load_model("lesion_CNN")
KNN_model = pickle.load(open('models/KNN.sav', 'rb'))
SVM_model = pickle.load(open('models/SVM.sav', 'rb'))
DT_model = pickle.load(open('models/DTC.sav', 'rb'))
ANN_model = pickle.load(open('models/MLP.sav', 'rb'))
Hybrid_model = pickle.load(open('models/Hybrid.sav', 'rb'))

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


def open_csv(filepath, filename):
    with open(filepath + filename, mode='r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            arr = lines

        data = list(arr)
        return data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'csv', 'bmp'}


# # upload function for length finding
# @app.route('/success', methods=['GET', 'POST'])
# def success():
#     global predictions, file_name, data, data_csv, answer
#     error = ''
#     target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')
#     if request.method == 'POST':
#
#         if request.files:
#             file = request.files['file']
#
#             if file and allowed_file(file.filename):
#
#                 file.save(os.path.join(target_img, file.filename))
#                 img_path = os.path.join(target_img, file.filename)
#                 file_name = file.filename
#
#                 pred_data(img_path)
#
#
#             else:
#                 error = "Please upload images_uploaded of jpg , jpeg and png extension only"
#
#         if len(error) == 0:
#             return render_template('results.html', img=file_name)
#         else:
#             return render_template('index.html', error=error)


@app.route('/success', methods=['GET', 'POST'])
def success():
    global img_path, file_name, predictions, predicted_class, confidence, predicted_class_CNN, predicted_class_SVM, predicted_class_KNN, predicted_class_DT, predicted_class_ANN, predicted_class_Hybrid, confidence_Hybrid, predicted_class_Hybrid2, confidence_Hybrid2, confidence_ANN, confidence_DT, confidence_KNN, confidence_SVM, confidence_CNN
    error = ''

    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        if request.files:
            file = request.files['file']

            if file and allowed_file(file.filename):

                img_path = os.path.join(target_img, file.filename)
                file.save(os.path.join(target_img, file.filename))

                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                file_name = file.filename

                processing(img_path)
                filteration(img_path)

            else:
                error = "Please upload images_uploaded of jpg , jpeg and png extension only"

        return render_template('preprocessing.html')


@app.route('/feature_extraction', methods=['GET', 'POST'])
def feature_extraction():
    global img_path, file_name, predictions, predicted_class, confidence, predicted_class_CNN, predicted_class_SVM, predicted_class_KNN, predicted_class_DT, predicted_class_ANN, predicted_class_Hybrid, confidence_Hybrid, predicted_class_Hybrid2, confidence_Hybrid2, confidence_ANN, confidence_DT, confidence_KNN, confidence_SVM, confidence_CNN
    error = ''

    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        if request.files:
            file = request.files['file']

            if file and allowed_file(file.filename):

                img_path = os.path.join(target_img, file.filename)
                file.save(os.path.join(target_img, file.filename))

                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                file_name = file.filename

                hsv(img_path)

            else:
                error = "Please upload images_uploaded of jpg , jpeg and png extension only"
    else:
        hsv(img_path)

    return render_template('feature_extraction.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global img_path, file_name, predictions, predicted_class, confidence, predicted_class_CNN, predicted_class_SVM, predicted_class_KNN, predicted_class_DT, predicted_class_ANN, predicted_class_Hybrid, confidence_Hybrid, predicted_class_Hybrid2, confidence_Hybrid2, confidence_ANN, confidence_DT, confidence_KNN, confidence_SVM, confidence_CNN
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        if request.files:
            file = request.files['file']

            if file and allowed_file(file.filename):

                img_path = os.path.join(target_img, file.filename)
                file.save(os.path.join(target_img, file.filename))

                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                file_name = file.filename

                processing(img_path)
                filteration(img_path)

            else:
                error = "Please upload images_uploaded of jpg , jpeg and png extension only"

    error = ''
    predicted_class_CNN, confidence_CNN = pred_and_plot(img_path, CNN_model)
    predicted_class_SVM, confidence_SVM = pred_ml_model(img_path, SVM_model)
    predicted_class_KNN, confidence_KNN = pred_ml_model(img_path, KNN_model)
    predicted_class_DT, confidence_DT = pred_ml_model(img_path, DT_model)
    predicted_class_ANN, confidence_ANN = pred_ml_model(img_path, ANN_model)
    predicted_class_Hybrid, confidence_Hybrid = pred_ml_model(img_path, Hybrid_model)

    if len(error) == 0:
        return render_template('results.html', img=file_name, type="img",
                               predicted_class_CNN=predicted_class_CNN, confidence_CNN=confidence_CNN,
                               predicted_class_SVM=predicted_class_SVM, confidence_SVM=confidence_SVM,
                               predicted_class_KNN=predicted_class_KNN, confidence_KNN=confidence_KNN,
                               predicted_class_DT=predicted_class_DT, confidence_DT=confidence_DT,
                               predicted_class_ANN=predicted_class_ANN, confidence_ANN=confidence_ANN,
                               predicted_class_Hybrid=predicted_class_Hybrid, confidence_Hybrid=confidence_Hybrid
                               )
    else:
        return render_template('index.html', error=error)


# ---------------------------------------- Metrics Display ------------------------------------------------------------

@app.route('/stats_CNN')
def stats_CNN():
    filename = 'stats_SVM.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_SVM.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_SVM.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="CNN", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_SVM')
def stats_SVM():
    filename = 'stats_SVM.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_SVM.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_SVM.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="SVM", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_KNN')
def stats_KNN():
    filename = 'stats_KNN.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_KNN.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_KNN.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="KNN", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_DT')
def stats_DT():
    filename = 'stats_DT.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_DT.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_DT.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="DT", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_ANN')
def stats_ANN():
    filename = 'stats_MLP.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_MLP.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_MLP.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="ANN", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_Hybrid')
def stats_Hybrid():
    filename = 'stats_Hybrid.csv'
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = 'static/assets/csv/stats_Hybrid.csv'
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    data_csv = pd.DataFrame(data_csv)

    img_name = 'stats_Hybrid.png'
    if ".csv" in filename:
        return render_template('algo_stats.html', algo="Hybrid", img=img_name,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


# ----------------------------------------------------------------------------------------------------


@app.route('/comparitive_analysis', methods=['GET', 'POST'])
def comparitive_analysis():
    # filename = 'compare.csv'
    # data = open_csv('static/assets/csv/', filename)

    compare_data = comparitive_plot(confidence_SVM,
                                    confidence_KNN,
                                    confidence_DT,
                                    confidence_ANN,
                                    confidence_CNN,
                                    confidence_Hybrid)

    # print(data)
    # f = 'static/assets/csv/compare.csv'
    # data_csv = []
    # with open(f) as file:
    #     csvfile = csv.reader(file)
    #     for row in csvfile:
    #         data_csv.append(row)
    #
    # data_csv = pd.DataFrame(data_csv)

    return render_template('comparitive_analysis.html',
                           compare_data=compare_data.to_html(classes='mystyle', index=False))


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == "__main__":
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(debug=True, port=2000)
