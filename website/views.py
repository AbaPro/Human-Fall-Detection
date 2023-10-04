from flask import Blueprint, render_template, request, flash, jsonify, Response
from flask_login import login_required, current_user
from .models import User
from .camera import gen_frames
from . import db  # means from __init__.py import db

# for versions
import platform
import sys
import flask
import flask_login
import flask_sqlalchemy
import tensorflow
import numpy
import cv2

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html", user=current_user)


@views.route('/about')
def about():

    return render_template("about.html", user=current_user,
                        platform_systm=platform.system(),
                        platform_version=platform.release(),
                        platform_architecture=platform.machine(),
                        platform_network=platform.node(),
                        platform_processor=platform.processor(),
                        python_version=sys.version,
                        flask_version=flask.__version__,
                        flask_sqlalchemy_version=flask_sqlalchemy.__version__,
                        flask_login_version=flask_login.__version__,
                        tensorflow_version=tensorflow.__version__,
                        numpy_version=numpy.__version__,
                        opencv_version=cv2.__version__
    )

@views.route('/start')
@login_required
def start():
    cameraOn = True
    return render_template("start.html", user=current_user, cameraOn=cameraOn)


@views.route('/users')
@login_required
def users():
    users = User.query.all()
    return render_template("users.html", user=current_user, users=users)


@views.route('/video_feed')
@login_required
def video_feed():

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
