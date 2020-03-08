import sqlite3

con = None
cur = None


def sqlite3_connect():
    global con

    con = sqlite3.connect('./image_db.db')


def sqlite3_create_tables():
    global cur

    cur = con.cursor()

    cur.execute('Create table IF NOT EXISTS image('
                'field_id INT PRIMARY KEY,'
                'image BLOB)')

    cur.execute('Create table IF NOT EXISTS image_data('
                'data_id DATA PRIMARY KEY,'
                'field_id INT,'
                'FOREIGN KEY (field_id) REFERENCES image (field_id))')

    cur.execute('Create table IF NOT EXISTS statistic_info('
                'statistic_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'field_id INTEGER,'
                'data_id DATA,'
                'cloud_rate REAL,'
                'index_weighted_avg REAL,'
                'standard_deviation REAL,'
                'unique_values_enter REAL,'
                'FOREIGN KEY (field_id) '
                'REFERENCES image (field_id)'
                'FOREIGN KEY (data_id)'
                'REFERENCES image_data (data_id))')


def sqlite3_insert_image(field_id, image):
    cur.execute('INSERT INTO image values(\'' + field_id + '\' , \'' + image + '\')')

    con.commit()


def sqlite3_insert_data(field_id, data_id):
    cur.execute('INSERT INTO image_data values(\'' + data_id + '\' , \'' + field_id + '\')')

    con.commit()


def sqlite3_insert_statistic(field_id, data_id, cloud_rate, index_weighted_avg, standard_deviation, unique_values_enter):
    cur.execute('INSERT INTO statistic_info(field_id, data_id, cloud_rate, index_weighted_avg, standard_deviation, unique_values_enter)'
                'values(\'' + field_id + '\' , \'' + data_id + '\' , \'' + cloud_rate + '\' , \''
                '' + index_weighted_avg + '\' , \'' + standard_deviation + '\' , \'' + unique_values_enter + '\')')

    con.commit()
