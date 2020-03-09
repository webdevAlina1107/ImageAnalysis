select image_id as Image_ID, field_id as Field_ID, data as Capture_date, image_data as Image_bitmap
from image where field_id = ? and data between ? and ?'