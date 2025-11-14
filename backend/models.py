# models.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class UploadedImage(db.Model):
    __tablename__ = 'uploaded_images'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)

    def __init__(self, filename, filepath):
        self.filename = filename
        self.filepath = filepath
