from flask_sqlalchemy import SQLAlchemy
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
db = SQLAlchemy()

class Object(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    id_culture = db.Column(db.Integer(), db.ForeignKey('cultures.id'), nullable=False)
    stage = db.Column(db.Integer())
    ref_name = db.Column(db.Text())

    Object = db.relationship('Cultures', backref='object', uselist=False)

    def __repr__(self):
        return f"<object {self.id}>"

class Cultures(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255))
    discription = db.Column(db.Text())
    modellink = db.Column(db.Text())
    clsofmodel = db.Column(db.Text())

    Cultures = db.relationship('Diseases', backref='cultures', uselist=False)
    Cultures1 = db.relationship('SolBioProt', backref='cultures', uselist=False)

    def __repr__(self):
        return f"<cultures {self.id}>"


class Diseases(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    cultur_id = db.Column(db.Integer(), db.ForeignKey('cultures.id'))
    name = db.Column(db.Text())
    discription = db.Column(db.Text())
    imagelink = db.Column(db.Text())
    num_dis = db.Column(db.Integer(), nullable=False)
    simptoms =  db.Column(db.Text())
    # Diseases = db.relationship('Parameters', backref='diseases', uselist=False)
    Diseases1 = db.relationship('Stage', backref='diseases', uselist=False)
    Diseases2 = db.relationship('SolBioProt', backref='diseases', uselist=False)

    def __repr__(self):
        return f"<diseases {self.id}>"


class Stage(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    id_dis = db.Column(db.Integer(), db.ForeignKey('diseases.id'))
    code = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    discription = db.Column(db.Text(), nullable=False)
    imagelink = db.Column(db.Text())

    def __repr__(self):
        return f"<stage {self.id}>"


class ProtectionBio(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.Text(), nullable=False)
    type = db.Column(db.Text(), nullable=False)

    ProtectionBio = db.relationship('SolBioProt', backref='protection_bio', uselist=False)

    def __repr__(self):
        return f"<protection_bio {self.id}>"


class SolBioProt(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    id_prot = db.Column(db.Integer(), db.ForeignKey('protection_bio.id'))
    id_dis = db.Column(db.Integer(), db.ForeignKey('diseases.id'))
    discription = db.Column(db.Text(), nullable=False)
    time_wait = db.Column(db.Text())
    time_to_work = db.Column(db.Text())
    id_cult = db.Column(db.Integer(), db.ForeignKey('cultures.id'))

    def __repr__(self):
        return f"<sol_bio_prot {self.id}>"