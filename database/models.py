from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=False, nullable=False)
    description = db.Column(db.Text, unique=False, nullable=False)

    def __repr__(self):
        return f'<User {self.description}>'
