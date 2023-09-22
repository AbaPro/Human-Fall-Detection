from flask import Blueprint, render_template, request, flash, jsonify,Response
from flask_login import login_required, current_user
from .models import User
from .camera import gen_frames
from . import db  # means from __init__.py import db

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html", user=current_user)

@views.route('/start')
@login_required
def start():
    cameraOn=True
    return render_template("start.html", user=current_user,cameraOn=cameraOn)

@views.route('/users')
@login_required
def users():
    users=User.query.all()
    return render_template("users.html", user=current_user , users=users)

@views.route('/video_feed')
@login_required
def video_feed():
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

