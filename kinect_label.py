import cv
import numpy as np
import json
import os
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QPoint, QTimer, Qt
from PyQt4.QtGui import QImage, QPainter, QWidget, QPen, QFileDialog
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Canvas(QtGui.QWidget):

    DELTA = 10

    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.draggin_idx = -1
        self.setGeometry(0,0,500,500)
        self.width = 10
        self.index = 0
        self.skipTimestamps = 0
        self.skippedFrame = 0
        self.maxFrameCount = 1
        self.xScale, self.yScale = [0,0]
        self.xMove, self.yMove = [0,0]
        self.original_points = np.array([[[0,0],[318, 0]],[[1,1], [1,1]]], dtype=np.float)
        self.points = self.original_points.copy()

    def paintEvent(self, e):
        if hasattr(self, 'points') and self.points.any():
            qp = QtGui.QPainter()
            qp.begin(self)
            self.drawPoints(qp)
            qp.end()

    def drawPoints(self, qp):
        qp.setPen(QPen(Qt.red, self.width))
        for point in self.points[self.index]:
            x, y = self.apply_point(point)
            qp.drawPoint(x,y)

    def _get_point(self, evt):
        return np.array(self._deapply_point([evt.pos().x(),evt.pos().y()]))

    def parse_csv(self, file_name, start_timestamp, frame_count):
        lines = [line.split(',') for line in open(file_name)]
        kinect_points = {}
        count = 0
        for line in lines:
            start_index = 1 if len(line) < 46 else 5
            if not all([n for n in line[start_index:-1]]) or float(line[start_index]) > 200:
                continue
            points = [float(n) for n in line[start_index:-1]]
            count+=1
            iter_points = iter(points)
            all_points = [[(320 - x)/320.0 * 318.0, y/240.0 * 198.0] for (x, y) in zip(iter_points, iter_points)]
            kinect_points[line[0]] = all_points #[all_points[i] for i in [3,4,5,7,8,9,11,12,13,14,16,17,18]]
        video_timestamps = [int(start_timestamp + 247/3.0 * i) for i in range(0, frame_count)]
        kinect_timestamps = np.array([int(timestamp) + self.skipTimestamps for timestamp in kinect_points.keys()],dtype=np.float)
        original_points = []
        for timestamp in video_timestamps:
            original_points.append(kinect_points.values()[np.argmin(abs(kinect_timestamps - timestamp))])
        self.original_points = np.array(original_points)
        self.points = self.original_points.copy()

    #get the click coordinates
    def mousePressEvent(self, evt):
        if evt.button() == QtCore.Qt.LeftButton and self.draggin_idx == -1:
            point = self._get_point(evt)
            dist = self.points[self.index] - point
            dist = dist[:,0]**2 + dist[:,1]**2
            dist[dist>self.DELTA] = np.inf
            if dist.min() < np.inf:
                self.draggin_idx = dist.argmin()

    def mouseMoveEvent(self, evt):
        if self.draggin_idx != -1:
            point = self._get_point(evt)
            dist = point - self.points[self.index][self.draggin_idx]
            self.original_points[self.index][self.draggin_idx] += dist
            self.points[self.index][self.draggin_idx] = point
            self.update()

    def mouseReleaseEvent(self, evt):
        if evt.button() == QtCore.Qt.LeftButton and self.draggin_idx != -1:
            point = self._get_point(evt)
            dist = point - self.points[self.index][self.draggin_idx]
            self.original_points[self.index][self.draggin_idx] += dist
            self.points[self.index][self.draggin_idx] = point
            self.draggin_idx = -1
            self.update()


    def rectOf(self, index, width):
        xRange = [self.apply_point(point)[0] for point in self.points[index] ]
        left, right = min(xRange), max(xRange)
        center = (left+right)/2.0

        return (int(center - width/2.0), 0, int(width), int(width))



    def scale(self, value, type = "x"):
        new_points = []
        if type == "x":
            self.xScale = value
        elif type == "y":
            self.yScale = value
        for point in self.original_points[self.index]:
            new_points.append(self.apply_point(point))
        self.points[self.index] = np.array(new_points)
        self.update()

    def move(self, value, type = "x"):
        new_points = []
        if type == "x":
            self.xMove = value
        elif type == "y":
            self.yMove = value
        for point in self.original_points[self.index]:
            new_points.append(self.apply_point(point))
        self.points[self.index] = np.array(new_points)
        self.update()

    def skipFrame(self, value):
        originalIndex = self.index - self.skippedFrame
        self.skippedFrame = value
        self.jumpTo(originalIndex)

    def skipTimestamp(self, value):
        self.skipTimestamps = value

    def jumpTo(self, value):
        self.index = int(value) + self.skippedFrame
        self.points[self.index] = [self.apply_point(point) for point in self.original_points[self.index]]
        self.update()
        pass

    def apply_point(self, point):
        x = point[0] * (100.0+self.xScale)/100.0 + self.xMove
        y = point[1] * (100.0+self.yScale)/100.0 + self.yMove
        return [x, y]

    def _deapply_point(self, point):
        x = (point[0] - self.xMove) / (100.0+self.xScale) * 100.0
        y = (point[1] - self.yMove) / (100.0+self.yScale) * 100.0
        return [x, y]

class IplQImage(QImage):
    """
    http://matthewshotton.wordpress.com/2011/03/31/python-opencv-iplimage-to-pyqt-qimage/
    A class for converting iplimages to qimages
    """

    def __init__(self, iplimage):
        # Rough-n-ready but it works dammit
        alpha = cv.CreateMat(iplimage.height, iplimage.width, cv.CV_8UC1)
        cv.Rectangle(alpha, (0, 0), (iplimage.width, iplimage.height), cv.ScalarAll(255), -1)
        rgba = cv.CreateMat(iplimage.height, iplimage.width, cv.CV_8UC4)
        cv.Set(rgba, (1, 2, 3, 4))
        cv.MixChannels([iplimage, alpha], [rgba], [
            (0, 0),  # rgba[0] -> bgr[2]
            (1, 1),  # rgba[1] -> bgr[1]
            (2, 2),  # rgba[2] -> bgr[0]
            (3, 3)  # rgba[3] -> alpha[0]
        ])
        self.__imagedata = rgba.tostring()
        super(IplQImage, self).__init__(self.__imagedata, iplimage.width, iplimage.height, QImage.Format_RGB32)


class VideoWidget(QWidget):
    """ A class for rendering video coming from OpenCV """

    def __init__(self, parent=None):
        QWidget.__init__(self)

    def init_video(self, file_name):
        self.file_name = file_name
        self._capture = cv.CaptureFromFile(file_name)
        # Take one frame to query height
        self.start_timestamp = int(file_name.split('/')[-1].split('.')[0])
        frame = cv.QueryFrame(self._capture)
        self.setMinimumSize(frame.width, frame.height)
        self.setMaximumSize(self.minimumSize())
        self.maxFrameCount = int(cv.GetCaptureProperty(self._capture, cv.CV_CAP_PROP_FRAME_COUNT))
        self.currentFrame = cv.GetCaptureProperty(self._capture, cv.CV_CAP_PROP_POS_FRAMES)
        self._frame = None
        self._image = self._build_image(frame)

    def play(self):
        self.queryFrame()
        # self._timer = QTimer(self)
        # self._timer.timeout.connect(self.queryFrame)
        # self._timer.start(1000.0/12.0)

    def export(self):
        capture = cv.CaptureFromFile(self.file_name)
        for i in range(0, self.maxFrameCount):
            frame = cv.QueryFrame(capture)
            path = os.path.join('results', str(self.start_timestamp))
            rect = self.canvas.rectOf(i, frame.height)
            points = []
            for point in self.canvas.original_points[i]:
                point = self.canvas.apply_point(point)
                points.append([point[0] - rect[0], point[1]])
            cv.SetImageROI(frame, rect)
            cv.SaveImage(os.path.join(path, str(i) + '.jpg'), frame)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, str(i) + '.txt'), 'wb+') as destination:
                destination.write(json.dumps(points))

    def _build_image(self, frame):
        if not self._frame:
            self._frame = cv.CreateImage((frame.width, frame.height), cv.IPL_DEPTH_8U, frame.nChannels)
        if frame.origin == cv.IPL_ORIGIN_TL:
            cv.Copy(frame, self._frame)
        else:
            cv.Flip(frame, self._frame, 0)
        return IplQImage(frame)

    def paintEvent(self, event):
        if hasattr(self, '_image') and self._image:
            painter = QPainter(self)
            painter.drawImage(QPoint(0, 0), self._image)

    def setSlider(self, slider):
        self.slider = slider

    def setCanvas(self, canvas):
        self.canvas = canvas
        self.canvas.maxFrameCount = self.maxFrameCount

    def queryFrame(self, byUser = False):
        frame = cv.QueryFrame(self._capture)
        self._image = self._build_image(frame)
        self.currentFrame = cv.GetCaptureProperty(self._capture, cv.CV_CAP_PROP_POS_FRAMES)
        if self.slider and not byUser:
            self.slider.setValue(self.currentFrame-1)
        if self.canvas:
            self.canvas.jumpTo(self.currentFrame-1)
        self.update()

    def jumpTo(self, value):
        cv.SetCaptureProperty(self._capture, cv.CV_CAP_PROP_POS_FRAMES, value)
        self.queryFrame(True)
        # self._timer.stop()
        self.update()



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.mainWindow = MainWindow
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(809, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 480, 791, 121))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.outputButton = QtGui.QPushButton(self.horizontalLayoutWidget)
        self.outputButton.setObjectName(_fromUtf8("outputButton"))
        self.horizontalLayout.addWidget(self.outputButton)
        self.openVideoButton = QtGui.QPushButton(self.horizontalLayoutWidget)
        self.openVideoButton.setObjectName(_fromUtf8("openVideoButton"))
        self.horizontalLayout.addWidget(self.openVideoButton)
        self.openCsvButton = QtGui.QPushButton(self.horizontalLayoutWidget)
        self.openCsvButton.setObjectName(_fromUtf8("openCsvButton"))
        self.horizontalLayout.addWidget(self.openCsvButton)
        self.playButton = QtGui.QPushButton(self.horizontalLayoutWidget)
        self.playButton.setObjectName(_fromUtf8("playButton"))
        self.horizontalLayout.addWidget(self.playButton)
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 231, 471))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.skipFrameLayout = QtGui.QHBoxLayout()
        self.skipFrameLayout.setObjectName(_fromUtf8("skipFrameLayout"))
        self.skipFrameLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.skipFrameLabel.setObjectName(_fromUtf8("skipFrameLabel"))
        self.skipFrameLayout.addWidget(self.skipFrameLabel)
        self.skipFrameEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.skipFrameEdit.setObjectName(_fromUtf8("skipFrameEdit"))
        self.skipFrameLayout.addWidget(self.skipFrameEdit)
        self.verticalLayout.addLayout(self.skipFrameLayout)
        self.skipTimestampLayout = QtGui.QHBoxLayout()
        self.skipTimestampLayout.setObjectName(_fromUtf8("skipTimestampLayout"))
        self.skipTimestampLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.skipTimestampLabel.setObjectName(_fromUtf8("skipTimestampLabel"))
        self.skipTimestampLayout.addWidget(self.skipTimestampLabel)
        self.skipTimestampEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.skipTimestampEdit.setObjectName(_fromUtf8("skipTimestampEdit"))
        self.skipTimestampLayout.addWidget(self.skipTimestampEdit)
        self.verticalLayout.addLayout(self.skipTimestampLayout)
        self.overallXLayout = QtGui.QHBoxLayout()
        self.overallXLayout.setObjectName(_fromUtf8("overallXLayout"))
        self.overallXLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.overallXLabel.setObjectName(_fromUtf8("overallXLabel"))
        self.overallXLayout.addWidget(self.overallXLabel)
        self.overallXEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.overallXEdit.setObjectName(_fromUtf8("overallXEdit"))
        self.overallXLayout.addWidget(self.overallXEdit)
        self.verticalLayout.addLayout(self.overallXLayout)
        self.overallYLayout = QtGui.QHBoxLayout()
        self.overallYLayout.setObjectName(_fromUtf8("overallYLayout"))
        self.overallYLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.overallYLabel.setObjectName(_fromUtf8("overallYLabel"))
        self.overallYLayout.addWidget(self.overallYLabel)
        self.overallYEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.overallYEdit.setObjectName(_fromUtf8("overallYEdit"))
        self.overallYLayout.addWidget(self.overallYEdit)
        self.verticalLayout.addLayout(self.overallYLayout)

        self.skeletonRateLayout = QtGui.QHBoxLayout()
        self.skeletonRateLayout.setObjectName(_fromUtf8("skeletonRateLayout"))
        self.skeletonRateLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.skeletonRateLabel.setObjectName(_fromUtf8("skeletonRateLabel"))
        self.skeletonRateLayout.addWidget(self.skeletonRateLabel)
        self.skeletonRateEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.skeletonRateEdit.setObjectName(_fromUtf8("skeletonRateEdit"))
        self.skeletonRateLayout.addWidget(self.skeletonRateEdit)
        self.verticalLayout.addLayout(self.skeletonRateLayout)

        self.scaleXLayout = QtGui.QVBoxLayout()
        self.scaleXLayout.setObjectName(_fromUtf8("scaleXLayout"))
        self.scaleXLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.scaleXLabel.setObjectName(_fromUtf8("scaleXLabel"))
        self.scaleXLayout.addWidget(self.scaleXLabel)
        self.xScaleSlider = QtGui.QSlider(self.verticalLayoutWidget)
        self.xScaleSlider.setOrientation(QtCore.Qt.Horizontal)
        self.xScaleSlider.setObjectName(_fromUtf8("xScaleSlider"))
        self.scaleXLayout.addWidget(self.xScaleSlider)
        self.verticalLayout.addLayout(self.scaleXLayout)
        self.scaleYLayout = QtGui.QVBoxLayout()
        self.scaleYLayout.setObjectName(_fromUtf8("scaleYLayout"))
        self.scaleYLabel = QtGui.QLabel(self.verticalLayoutWidget)
        self.scaleYLabel.setObjectName(_fromUtf8("scaleYLabel"))
        self.scaleYLayout.addWidget(self.scaleYLabel)
        self.yScaleSlider = QtGui.QSlider(self.verticalLayoutWidget)
        self.yScaleSlider.setOrientation(QtCore.Qt.Horizontal)
        self.yScaleSlider.setObjectName(_fromUtf8("yScaleSlider"))
        self.scaleYLayout.addWidget(self.yScaleSlider)
        self.verticalLayout.addLayout(self.scaleYLayout)
        self.videoController = QtGui.QWidget(self.centralwidget)
        self.videoController.setGeometry(QtCore.QRect(250, 10, 551, 461))
        self.videoController.setObjectName(_fromUtf8("videoController"))
        self.videoLayout = QtGui.QVBoxLayout(self.videoController)
        self.videoLayout.setObjectName(_fromUtf8("videoLayout"))
        self.videoView = VideoWidget(self.videoController)
        self.videoView.setObjectName(_fromUtf8("videoView"))
        self.canvas = Canvas(self.videoView)
        self.canvas.setObjectName(_fromUtf8("canvas"))
        self.videoLayout.addWidget(self.videoView)
        self.videoSlider = QtGui.QSlider(self.videoController)
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setObjectName(_fromUtf8("videoSlider"))
        self.videoLayout.addWidget(self.videoSlider)
        self.horizontalLayoutWidget.raise_()
        self.verticalLayoutWidget.raise_()
        self.scaleYLabel.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.connectActions()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.outputButton.setText(_translate("MainWindow", "output", None))
        self.openVideoButton.setText(_translate("MainWindow", "open video", None))
        self.openCsvButton.setText(_translate("MainWindow", "open csv", None))
        self.playButton.setText(_translate("MainWindow", "play", None))
        self.skipFrameLabel.setText(_translate("MainWindow", "skip frame", None))
        self.skipTimestampLabel.setText(_translate("MainWindow", "start timestamp", None))
        self.overallXLabel.setText(_translate("MainWindow", "overall x", None))
        self.overallYLabel.setText(_translate("MainWindow", "overall y", None))
        self.skeletonRateLabel.setText(_translate("MainWindow", "skeleton rate", None))
        self.scaleXLabel.setText(_translate("MainWindow", "scale x", None))
        self.scaleYLabel.setText(_translate("MainWindow", "scale y", None))

    def connectActions(self):
        self.xScaleSlider.valueChanged.connect(self.xScaled)
        self.yScaleSlider.valueChanged.connect(self.yScaled)
        self.videoSlider.valueChanged.connect(self.videoSlide)
        self.overallXEdit.textChanged.connect(self.overallXMoved)
        self.overallYEdit.textChanged.connect(self.overallYMoved)
        self.skipFrameEdit.textChanged.connect(self.skipFrame)
        self.skipTimestampEdit.textChanged.connect(self.skipTimestamp)

        self.openCsvButton.clicked.connect(self.openCsvFiles)
        self.openVideoButton.clicked.connect(self.openVideoFiles)
        self.playButton.clicked.connect(self.playVideo)
        self.outputButton.clicked.connect(self.exportData)

    def playVideo(self):
        self.videoView.play()

    def exportData(self):
        self.videoView.export()

    def openCsvFiles(self):
        if hasattr(self.videoView,"canvas"):
            file_name = self.openFiles()
            self.canvas.parse_csv(file_name, self.videoView.start_timestamp, self.videoView.maxFrameCount)
        else:
            QtGui.QMessageBox.about(MainWindow, 'Error','Select Video First')

    def openVideoFiles(self):
        file_name = self.openFiles()
        self.videoView.init_video(file_name)
        self.videoView.setSlider(self.videoSlider)
        self.videoView.setCanvas(self.canvas)
        self.videoSlider.setMinimum(self.videoView.currentFrame)
        self.videoSlider.setMaximum(self.videoView.maxFrameCount)

    def openFiles(self):
        files = QFileDialog.getOpenFileName(self.mainWindow, 'Open File',".")
        return str(files)

    def xScaled(self, value):
        self.canvas.scale(value, "x")

    def yScaled(self, value):
        self.canvas.scale(value, "y")

    def overallXMoved(self, value):
        number = self.getNumber(value)
        if number:
            self.canvas.move(number, "x")

    def overallYMoved(self, value):
        number = self.getNumber(value)
        if number:
            self.canvas.move(number, "y")

    def skipFrame(self, value):
        number = self.getNumber(value)
        if number:
            self.canvas.skipFrame(number)

    def skipTimestamp(self, value):
        number = self.getNumber(value)
        if number:
            self.canvas.skipTimestamp(number)

    def videoSlide(self, value):
        self.videoView.jumpTo(value)

    def getNumber(self, value):
        try:
            return int(value) if value and value != '-' else 0
        except Exception:
            QtGui.QMessageBox.about(MainWindow, 'Error','Input can only be an integer')
            pass


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
