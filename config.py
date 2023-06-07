class Config(object):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///StoreDB.sqlite3"
    SQLALCHEMY_TRACK_MODIFICATION = False
    UPLOAD_FOLDER = "./static/images"


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'mysql://user@localhost/foo'


class DevelopmentConfig(Config):
    DEBUG = True
    SECRET_KEY = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
    #DATABASE_URI = 'sqlite://:memory:'


class TestConfig(Config):
    Testing = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///test_DB.sqlite3"
    SQLALCHEMY_TRACK_MODIFICATION = False