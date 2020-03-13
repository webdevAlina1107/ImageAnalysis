select cloud_rate, index_weighted_avg, standard_deviation, confidence_interval_lower, confidence_interval_upper,
capture_date, image.image_id as image_id from statistic_info inner join image on
 (statistic_info.image_id = image.image_id) where image.field_id = ? and image.capture_date between ? and ?
 and statistic_info.cloud_rate < ? order by image.capture_date asc