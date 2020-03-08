import sqlite3

con = None
cur = None


def sqlite3_connect():
    global con
    global cur
    con = sqlite3.connect('.db/datasource/images.db')
    cur = con.cursor()


def sqlite3_import_images(image_id, field_id, image_data, data):
    cur.execute(f"INSERT INTO image VALUES ('{image_id}', '{field_id}', {image_data}', '{data}')")
    cur.commit()


# import image values

def sqlite3_import_field(field_id):
    cur.execute(f"INSERT INTO field VALUES ('{field_id})")
    cur.commit()


# import field from file to db

def sqlite3_export_image(field_id, data_start, data_end):
    cur.execute(f"SELECT * FROM image, field where image.field_id = field.field_id and field.field_id = '{field_id}'"
                f"and data between '{data_start}' and '{data_end}'")
    images = cur.fetchall()
    return images


# get image from db (date to date)

def sqlite3_export_image_all():
    cur.execute(f"SELECT * FROM image, field where image.field_id = field.field_id")
    images = cur.fetchall()
    return images


# get image all from db


def sqlite3_filter_by_date_image(data_start, data_end):
    cur.execute(
        f"Select * from image, field where image.field_id = field.field_id and data between '{data_start}' and '{data_end}'")
    images = cur.fetchall()
    return images


# filter by date


def sqlite3_filter_by_date_head(head_size, data_start, data_end):
    cur.execute(
        "Select * from image, field where image.field_id = field.field_id and data between \'"
        f" '{data_start}'  and '{data_end}' LIMIT {head_size} ")
    images = cur.fetchmany(head_size)
    return images


# filter by head

def sqlite3_visualize_statistic(id, start, end, max_cloudiness):
    cur.execute(
        "Select data, index_weighted_avg, confidence_interval_lower, confidence_interval_upper from statistic_info, image, field"
        "where statistic_info.image_id = image.image_id and field.field_id = image.field_id"
        f"and data between '{start}' and '{end} ' and cloud_rate < '{max_cloudiness}' and field.field_id = '{id}' order by data asc")
    data = cur.fetchall()
    return data


# get statistic info

def sqlite3_visualize_clouds(id, start, end):
    cur.execute("Select cloud_rate from image, field, statistic_info where field.field_id = image.field_id "
                f"and image.image_id = statistic_info.image_id and field.field_id = '{id}' and data between '{start}' and '{end}'")
    data = cur.fetchall()
    return data

# get cloud
